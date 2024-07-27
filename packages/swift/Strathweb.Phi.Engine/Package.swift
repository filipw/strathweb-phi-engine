// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "Strathweb.Phi.Engine",
    platforms: [
        .iOS(.v15), .macOS(.v14)
    ],
    products: [
        .library(
            name: "Strathweb.Phi.Engine",
            targets: ["Strathweb.Phi.Engine"]),
    ],
    targets: [
        .target(
            name: "Strathweb.Phi.Engine",
            dependencies: ["Strathweb.Phi.Engine.FFI"],
            path: "Sources/Strathweb.Phi.Engine",
            linkerSettings: [
                .linkedFramework("Metal", .when(platforms: [.macOS])),
                .linkedFramework("MetalPerformanceShaders", .when(platforms: [.macOS]))
            ]),
        .target(
            name: "Strathweb.Phi.Engine.FFI",
            dependencies: ["strathweb_phi_engine_framework"],
            path: "Sources/FFI",
            publicHeadersPath: "include"),
        .binaryTarget(
            name: "strathweb_phi_engine_framework",
            path: "Libs/strathweb_phi_engine_framework.xcframework"),
    ]
)