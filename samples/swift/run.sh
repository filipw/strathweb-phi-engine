#!/bin/bash

rm -rf .build

cd ../../
./build-swift.sh
cd samples/swift/sample-executable

swift run -c release