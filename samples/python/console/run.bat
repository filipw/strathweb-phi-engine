@echo off

del /f strathweb_phi_engine.dll
del /f strathweb_phi_engine.py

cargo build --release --manifest-path ..\..\..\strathweb-phi-engine\Cargo.toml

copy ..\..\..\strathweb-phi-engine\target\release\strathweb_phi_engine.dll .
copy ..\..\..\strathweb-phi-engine\bindings\strathweb_phi_engine.py .

python main.py