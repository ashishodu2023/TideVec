from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMake, cmake_layout
from conan.tools.files import copy
import os

class TideVecConan(ConanFile):
    name        = "tidevec"
    version     = "0.1.0"
    license     = "Apache-2.0"
    author      = "Ashish Verma <contact@getcortexops.com>"
    url         = "https://github.com/ashishodu2023/TideVec"
    homepage    = "https://tidevec.com"
    description = "Temporally-aware causal vector database — C++20 SDK"
    topics      = ("vector-database", "embeddings", "ANN", "temporal", "causal")
    settings    = "os", "compiler", "build_type", "arch"
    no_copy_source = True

    def layout(self):
        cmake_layout(self, src_folder="sdk/cpp")

    def package_id(self):
        # Header-only — no binary variation
        self.info.clear()

    def package(self):
        # Copy all headers
        copy(self, "*.hpp",
             src=os.path.join(self.source_folder, "include"),
             dst=os.path.join(self.package_folder, "include"))
        # Copy license
        copy(self, "LICENSE",
             src=os.path.join(self.source_folder, "..", ".."),
             dst=os.path.join(self.package_folder, "licenses"))

    def package_info(self):
        self.cpp_info.bindirs  = []
        self.cpp_info.libdirs  = []
        self.cpp_info.set_property("cmake_file_name",   "tidevec")
        self.cpp_info.set_property("cmake_target_name", "tidevec::tidevec")
