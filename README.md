# flatppl-bridge

Serves black-box models to FlatPPL engines. A black box can be a log density, a sampler or an expensive function, implemented by BridgeStan, a compiled library, an executable or Python code.

A FlatPPL model declares a box as a module of `external(anything)` members and queries it with the usual `logdensityof`, `rand`, `~` and function application. The engine sends the box only primitive queries over HTTP, with data as Arrow IPC.

The server core is Rust. Adapters are C ABI plugins, and a Python layer sits on top.

Work in progress.
