// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "SampleExecutable",
    platforms: [
        .macOS(.v14)
    ],
    dependencies: [
        .package(path: "../../../packages/swift/Strathweb.Phi.Engine")
    ],
    targets: [
        .executableTarget(
            name: "SampleExecutable",
            dependencies: [
                .product(name: "Strathweb.Phi.Engine", package: "Strathweb.Phi.Engine")
            ],
            path: "Sources/SampleExecutable",
        )
    ]
)