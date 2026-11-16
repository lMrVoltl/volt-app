plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}
android {
    namespace = "com.voltage.ai"
    compileSdk = 34
    defaultConfig {
        applicationId = "com.voltage.ai"
        minSdk = 26
        targetSdk = 34
        versionCode = 4
        versionName = "1.1.1"
        ndk { abiFilters += listOf("arm64-v8a") }
        externalNativeBuild {
            cmake {
                arguments += listOf(
                    "-DANDROID_STL=c++_shared",
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DGGML_NATIVE=OFF",
                    "-DGGML_CPU_KLEIDIAI=ON"
                )
            }
        }
    }
    signingConfigs {
        create("voltSign") {
            storeFile = file("volt.keystore")
            storePassword = "voltage123"
            keyAlias = "volt"
            keyPassword = "voltage123"
        }
    }
    buildTypes {
        debug {
            signingConfig = signingConfigs.getByName("voltSign")
            externalNativeBuild {
                cmake {
                    arguments += listOf("-DCMAKE_BUILD_TYPE=Release")
                }
            }
        }
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("voltSign")
        }
    }
    externalNativeBuild {
        cmake { path = file("src/main/cpp/CMakeLists.txt") }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
}
dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.webkit:webkit:1.11.0")
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation("com.google.mlkit:text-recognition:16.0.1")
}
