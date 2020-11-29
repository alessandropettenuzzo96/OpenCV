// swift-tools-version:5.3
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "OpenCV",
    platforms: [
        .iOS(.v10)
    ],
    products: [
        // Products define the executables and libraries a package produces, and make them visible to other packages.
        .library(
            name: "OpenCV",
            targets: ["OpenCV"]),
    ],
    dependencies: [
        // Dependencies declare other packages that this package depends on.
        // .package(url: /* package url */, from: "1.0.0"),
    ],
    targets: [
        // Targets are the basic building blocks of a package. A target can define a module or a test suite.
        // Targets can depend on other targets in this package, and on products in packages this package depends on.
        .target(
            name: "OpenCV",
            dependencies: [
                "opencv2"
            ],
            exclude: [
                "opencv2.xcframework"
            ]
        ),
        .binaryTarget(
            name: "opencv2",
            path: "./Sources/OpenCV/opencv2.xcframework"
        ),
    ],
    cLanguageStandard: CLanguageStandard.c11,
    cxxLanguageStandard: .cxx11
)
