// -------------------------------------------------------------------------------------------------
//  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
//  https://nautechsystems.io
//
//  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
//  You may not use this file except in compliance with the License.
//  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
// -------------------------------------------------------------------------------------------------

use std::{collections::HashMap, sync::Arc};

use nautilus_core::consts::NAUTILUS_USER_AGENT;
use nautilus_model::defi::{Block, Chain, rpc::RpcNodeWssResponse};
use nautilus_network::websocket::{Consumer, WebSocketClient, WebSocketConfig};
use reqwest::header::USER_AGENT;
use tokio_tungstenite::tungstenite::Message;

use crate::rpc::{
    error::BlockchainRpcClientError,
    types::{BlockchainMessage, RpcEventType},
    utils::{
        extract_rpc_subscription_id, is_subscription_confirmation_response, is_subscription_event,
    },
};

/// Core implementation of a blockchain RPC client that serves as the base for all chain-specific clients.
///
/// It provides a shared implementation of common blockchain RPC functionality, handling:
/// - WebSocket connection management with blockchain RPC node.
/// - Subscription lifecycle (creation, tracking, and termination).
/// - Message serialization and deserialization of RPC messages.
/// - Event type mapping and dispatching.
#[derive(Debug)]
pub struct CoreBlockchainRpcClient {
    /// The blockchain network type this client connects to.
    chain: Chain,
    /// WebSocket secure URL for the blockchain node's RPC endpoint.
    wss_rpc_url: String,
    /// Auto-incrementing counter for generating unique RPC request IDs.
    request_id: u64,
    /// Tracks in-flight subscription requests by mapping request IDs to their event types.
    pending_subscription_request: HashMap<u64, RpcEventType>,
    /// Maps active subscription IDs to their corresponding event types for message
    /// deserialization.
    subscription_event_types: HashMap<String, RpcEventType>,
    /// The active WebSocket client connection.
    wss_client: Option<Arc<WebSocketClient>>,
    /// Channel receiver for consuming WebSocket messages.
    wss_consumer_rx: Option<tokio::sync::mpsc::Receiver<Message>>,
}

impl CoreBlockchainRpcClient {
    #[must_use]
    pub fn new(chain: Chain, wss_rpc_url: String) -> Self {
        Self {
            chain,
            wss_rpc_url,
            request_id: 1,
            wss_client: None,
            pending_subscription_request: HashMap::new(),
            subscription_event_types: HashMap::new(),
            wss_consumer_rx: None,
        }
    }

    /// Establishes a WebSocket connection to the blockchain node and sets up the message channel.
    ///
    /// # Errors
    ///
    /// Returns an error if the WebSocket connection fails.
    pub async fn connect(&mut self) -> anyhow::Result<()> {
        let (tx, rx) = tokio::sync::mpsc::channel(100);
        let user_agent = (USER_AGENT.to_string(), NAUTILUS_USER_AGENT.to_string());
        // Most of the blockchain rpc nodes require a heartbeat to keep the connection alive
        let heartbeat_interval = 30;
        let config = WebSocketConfig {
            url: self.wss_rpc_url.clone(),
            headers: vec![user_agent],
            handler: Consumer::Rust(tx),
            heartbeat: Some(heartbeat_interval),
            heartbeat_msg: None,
            #[cfg(feature = "python")]
            ping_handler: None,
            reconnect_timeout_ms: Some(5_000),
            reconnect_delay_initial_ms: None,
            reconnect_delay_max_ms: None,
            reconnect_backoff_factor: None,
            reconnect_jitter_ms: None,
        };
        let client = WebSocketClient::connect(
            config,
            #[cfg(feature = "python")]
            None,
            #[cfg(feature = "python")]
            None,
            #[cfg(feature = "python")]
            None,
            vec![],
            None,
        )
        .await?;

        self.wss_client = Some(Arc::new(client));
        self.wss_consumer_rx = Some(rx);

        Ok(())
    }

    /// Registers a subscription for the specified event type and records it internally with the given ID.
    async fn subscribe_events(
        &mut self,
        event_type: RpcEventType,
        subscription_id: String,
    ) -> Result<(), BlockchainRpcClientError> {
        if let Some(client) = &self.wss_client {
            log::info!("Subscribing to new blocks on chain '{}'", self.chain.name);
            let msg = serde_json::json!({
                "method": "eth_subscribe",
                "id": self.request_id,
                "jsonrpc": "2.0",
                "params": [subscription_id]
            });
            self.pending_subscription_request
                .insert(self.request_id, event_type);
            self.request_id += 1;
            if let Err(err) = client.send_text(msg.to_string(), None).await {
                log::error!("Error sending subscribe message: {err:?}");
            }
            Ok(())
        } else {
            Err(BlockchainRpcClientError::ClientError(String::from(
                "Client not connected",
            )))
        }
    }

    /// Terminates a subscription with the blockchain node using the provided subscription ID.
    async fn unsubscribe_events(
        &self,
        subscription_id: String,
    ) -> Result<(), BlockchainRpcClientError> {
        if let Some(client) = &self.wss_client {
            log::info!("Unsubscribing to new blocks on chain {}", self.chain.name);
            let msg = serde_json::json!({
                "method": "eth_unsubscribe",
                "id": 1,
                "jsonrpc": "2.0",
                "params": [subscription_id]
            });
            if let Err(err) = client.send_text(msg.to_string(), None).await {
                log::error!("Error sending unsubscribe message: {err:?}");
            }
            Ok(())
        } else {
            Err(BlockchainRpcClientError::ClientError(String::from(
                "Client not connected",
            )))
        }
    }

    /// Waits for and returns the next available message from the WebSocket channel.
    pub async fn wait_on_rpc_channel(&mut self) -> Option<Message> {
        match &mut self.wss_consumer_rx {
            Some(rx) => rx.recv().await,
            None => None,
        }
    }

    /// Retrieves, parses, and returns the next blockchain RPC message as a structured `BlockchainRpcMessage` type.
    ///
    /// # Panics
    ///
    /// Panics if expected fields (`id`, `result`) are missing or cannot be converted when handling subscription confirmations or events.
    ///
    /// # Errors
    ///
    /// Returns an error if the RPC channel encounters an error or if deserialization of the message fails.
    pub async fn next_rpc_message(
        &mut self,
    ) -> Result<BlockchainMessage, BlockchainRpcClientError> {
        while let Some(msg) = self.wait_on_rpc_channel().await {
            match msg {
                Message::Text(text) => match serde_json::from_str::<serde_json::Value>(&text) {
                    Ok(json) => {
                        if is_subscription_confirmation_response(&json) {
                            let subscription_request_id = json.get("id").unwrap().as_u64().unwrap();
                            let result = json.get("result").unwrap().as_str().unwrap();
                            let event_type = self
                                .pending_subscription_request
                                .get(&subscription_request_id)
                                .unwrap();
                            self.subscription_event_types
                                .insert(result.to_string(), event_type.clone());
                            self.pending_subscription_request
                                .remove(&subscription_request_id);
                            continue;
                        } else if is_subscription_event(&json) {
                            let subscription_id = match extract_rpc_subscription_id(&json) {
                                Some(id) => id,
                                None => {
                                    return Err(BlockchainRpcClientError::InternalRpcClientError(
                                        "Error parsing subscription id from valid rpc response"
                                            .to_string(),
                                    ));
                                }
                            };
                            if let Some(event_type) =
                                self.subscription_event_types.get(subscription_id)
                            {
                                match event_type {
                                    RpcEventType::NewBlock => {
                                        return match serde_json::from_value::<
                                            RpcNodeWssResponse<Block>,
                                        >(json)
                                        {
                                            Ok(block_response) => {
                                                let block = block_response.params.result;
                                                Ok(BlockchainMessage::Block(block))
                                            }
                                            Err(e) => {
                                                Err(BlockchainRpcClientError::MessageParsingError(
                                                    format!(
                                                        "Error parsing rpc response to block with error {e}"
                                                    ),
                                                ))
                                            }
                                        };
                                    }
                                }
                            }
                            return Err(BlockchainRpcClientError::InternalRpcClientError(format!(
                                "Event type not found for defined subscription id {subscription_id}"
                            )));
                        }
                        return Err(BlockchainRpcClientError::UnsupportedRpcResponseType(
                            json.to_string(),
                        ));
                    }
                    Err(e) => {
                        return Err(BlockchainRpcClientError::MessageParsingError(e.to_string()));
                    }
                },
                Message::Pong(_) => {
                    continue;
                }
                _ => {
                    return Err(BlockchainRpcClientError::UnsupportedRpcResponseType(
                        msg.to_string(),
                    ));
                }
            }
        }

        Err(BlockchainRpcClientError::NoMessageReceived)
    }

    /// Subscribes to real-time block updates from the blockchain node.
    ///
    /// # Errors
    ///
    /// Returns an error if the subscription request fails or if the client is not connected.
    pub async fn subscribe_blocks(&mut self) -> Result<(), BlockchainRpcClientError> {
        self.subscribe_events(RpcEventType::NewBlock, String::from("newHeads"))
            .await
    }

    /// Cancels the subscription to real-time block updates.
    ///
    /// # Errors
    ///
    /// Returns an error if the unsubscription request fails or if the client is not connected.
    pub async fn unsubscribe_blocks(&mut self) -> Result<(), BlockchainRpcClientError> {
        self.unsubscribe_events(String::from("newHeads")).await?;

        // Find and remove the subscription ID associated with the newBlock event type
        let subscription_ids_to_remove: Vec<String> = self
            .subscription_event_types
            .iter()
            .filter(|(_, event_type)| **event_type == RpcEventType::NewBlock)
            .map(|(id, _)| id.clone())
            .collect();

        for id in subscription_ids_to_remove {
            self.subscription_event_types.remove(&id);
        }
        Ok(())
    }
}
