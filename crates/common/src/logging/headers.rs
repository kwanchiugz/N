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

use nautilus_core::UUID4;
use nautilus_model::identifiers::TraderId;
use sysinfo::System;
use ustr::Ustr;

use crate::{enums::LogColor, logging::log_info};

#[rustfmt::skip]
pub fn log_header(trader_id: TraderId, machine_id: &str, instance_id: UUID4, component: Ustr) {
    let mut sys = System::new_all();
    sys.refresh_all();

    let c = component;

    let kernel_version = System::kernel_version().map_or(String::new(), |v| format!("kernel-{v} "));
    let os_version = System::long_os_version().unwrap_or_default();
    let pid = std::process::id();

    header_sepr(c, "=================================================================");
    header_sepr(c, " NAUTILUS TRADER - Automated Algorithmic Trading Platform");
    header_sepr(c, " by Nautech Systems Pty Ltd.");
    header_sepr(c, " Copyright (C) 2015-2025. All rights reserved.");
    header_sepr(c, "=================================================================");
    header_line(c, "");
    header_line(c, "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣴⣶⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀");
    header_line(c, "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⣾⣿⣿⣿⠀⢸⣿⣿⣿⣿⣶⣶⣤⣀⠀⠀⠀⠀⠀");
    header_line(c, "⠀⠀⠀⠀⠀⠀⢀⣴⡇⢀⣾⣿⣿⣿⣿⣿⠀⣾⣿⣿⣿⣿⣿⣿⣿⠿⠓⠀⠀⠀⠀");
    header_line(c, "⠀⠀⠀⠀⠀⣰⣿⣿⡀⢸⣿⣿⣿⣿⣿⣿⠀⣿⣿⣿⣿⣿⣿⠟⠁⣠⣄⠀⠀⠀⠀");
    header_line(c, "⠀⠀⠀⠀⢠⣿⣿⣿⣇⠀⢿⣿⣿⣿⣿⣿⠀⢻⣿⣿⣿⡿⢃⣠⣾⣿⣿⣧⡀⠀⠀");
    header_line(c, "⠀⠀⠀⠠⣾⣿⣿⣿⣿⣿⣧⠈⠋⢀⣴⣧⠀⣿⡏⢠⡀⢸⣿⣿⣿⣿⣿⣿⣿⡇⠀");
    header_line(c, "⠀⠀⠀⣀⠙⢿⣿⣿⣿⣿⣿⠇⢠⣿⣿⣿⡄⠹⠃⠼⠃⠈⠉⠛⠛⠛⠛⠛⠻⠇⠀");
    header_line(c, "⠀⠀⢸⡟⢠⣤⠉⠛⠿⢿⣿⠀⢸⣿⡿⠋⣠⣤⣄⠀⣾⣿⣿⣶⣶⣶⣦⡄⠀⠀⠀");
    header_line(c, "⠀⠀⠸⠀⣾⠏⣸⣷⠂⣠⣤⠀⠘⢁⣴⣾⣿⣿⣿⡆⠘⣿⣿⣿⣿⣿⣿⠀⠀⠀⠀");
    header_line(c, "⠀⠀⠀⠀⠛⠀⣿⡟⠀⢻⣿⡄⠸⣿⣿⣿⣿⣿⣿⣿⡀⠘⣿⣿⣿⣿⠟⠀⠀⠀⠀");
    header_line(c, "⠀⠀⠀⠀⠀⠀⣿⠇⠀⠀⢻⡿⠀⠈⠻⣿⣿⣿⣿⣿⡇⠀⢹⣿⠿⠋⠀⠀⠀⠀⠀");
    header_line(c, "⠀⠀⠀⠀⠀⠀⠋⠀⠀⠀⡘⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠁⠀⠀⠀⠀⠀⠀⠀");
    header_line(c, "");
    header_sepr(c, "=================================================================");
    header_sepr(c, " SYSTEM SPECIFICATION");
    header_sepr(c, "=================================================================");
    header_line(c, &format!("CPU architecture: {}", sys.cpus()[0].brand()));
    header_line(c, &format!("CPU(s): {} @ {} Mhz", sys.cpus().len(), sys.cpus()[0].frequency()));
    header_line(c, &format!("OS: {kernel_version}{os_version}"));

    log_sysinfo(component);

    header_sepr(c, "=================================================================");
    header_sepr(c, " IDENTIFIERS");
    header_sepr(c, "=================================================================");
    header_line(c, &format!("trader_id: {trader_id}"));
    header_line(c, &format!("machine_id: {machine_id}"));
    header_line(c, &format!("instance_id: {instance_id}"));
    header_line(c, &format!("PID: {pid}"));

    header_sepr(c, "=================================================================");
    header_sepr(c, " VERSIONING");
    header_sepr(c, "=================================================================");

    #[cfg(not(feature = "python"))]
    log_rust_versioning(c);

    #[cfg(feature = "python")]
    log_python_versioning(c);
}

#[cfg(not(feature = "python"))]
#[rustfmt::skip]
fn log_rust_versioning(c: Ustr) {
    use nautilus_core::consts::NAUTILUS_VERSION;
    header_line(c, &format!("nautilus_trader: {NAUTILUS_VERSION}"));
}

#[cfg(feature = "python")]
#[rustfmt::skip]
fn log_python_versioning(c: Ustr) {
    if !python_available() {
        return;
    }

    let package = "nautilus_trader";
    header_line(c, &format!("{package}: {}", python_package_version(package)));
    header_line(c, &format!("python: {}", python_version()));

    for package in ["numpy", "pandas", "msgspec", "pyarrow", "pytz", "uvloop"] {
        header_line(c, &format!("{package}: {}", python_package_version(package)));
    }

    header_sepr(c, "=================================================================");
}

#[cfg(feature = "python")]
#[inline]
#[allow(unsafe_code)]
fn python_available() -> bool {
    // SAFETY: `Py_IsInitialized` reads a flag and is safe to call at any
    // time, even before the interpreter has been started.
    unsafe { pyo3::ffi::Py_IsInitialized() != 0 }
}

#[rustfmt::skip]
pub fn log_sysinfo(component: Ustr) {
    let mut sys = System::new_all();
    sys.refresh_all();

    let c = component;

    let ram_total = sys.total_memory();
    let ram_used = sys.used_memory();
    let ram_used_p = (ram_used as f64 / ram_total as f64) * 100.0;
    let ram_avail = ram_total - ram_used;
    let ram_avail_p = (ram_avail as f64 / ram_total as f64) * 100.0;

    let swap_total = sys.total_swap();
    let swap_used = sys.used_swap();
    let swap_used_p = (swap_used as f64 / swap_total as f64) * 100.0;
    let swap_avail = swap_total - swap_used;
    let swap_avail_p = (swap_avail as f64 / swap_total as f64) * 100.0;

    header_sepr(c, "=================================================================");
    header_sepr(c, " MEMORY USAGE");
    header_sepr(c, "=================================================================");
    header_line(c, &format!("RAM-Total: {:.2} GiB", bytes_to_gib(ram_total)));
    header_line(c, &format!("RAM-Used: {:.2} GiB ({:.2}%)", bytes_to_gib(ram_used), ram_used_p));
    header_line(c, &format!("RAM-Avail: {:.2} GiB ({:.2}%)", bytes_to_gib(ram_avail), ram_avail_p));
    header_line(c, &format!("Swap-Total: {:.2} GiB", bytes_to_gib(swap_total)));
    header_line(c, &format!("Swap-Used: {:.2} GiB ({:.2}%)", bytes_to_gib(swap_used), swap_used_p));
    header_line(c, &format!("Swap-Avail: {:.2} GiB ({:.2}%)", bytes_to_gib(swap_avail), swap_avail_p));
}

fn header_sepr(c: Ustr, s: &str) {
    log_info!("{}", s, color = LogColor::Cyan, component = c.as_str());
}

fn header_line(c: Ustr, s: &str) {
    log_info!("{}", s, component = c.as_str());
}

fn bytes_to_gib(b: u64) -> f64 {
    b as f64 / (2u64.pow(30) as f64)
}

#[cfg(feature = "python")]
fn python_package_version(package: &str) -> String {
    nautilus_core::python::version::get_python_package_version(package)
}

#[cfg(feature = "python")]
fn python_version() -> String {
    nautilus_core::python::version::get_python_version()
}
