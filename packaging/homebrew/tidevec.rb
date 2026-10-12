# Homebrew formula for the TideVec server.
#
# Lives in a tap (e.g. github.com/ashishodu2023/homebrew-tap as
# Formula/tidevec.rb) until it qualifies for homebrew-core:
#
#   brew install ashishodu2023/tap/tidevec
#   brew services start tidevec
#
# On release, packaging/homebrew/update_formula.sh rewrites url/sha256.
class Tidevec < Formula
  desc "Vector database that knows when facts change (temporal decay, supersession)"
  homepage "https://tidevec.com"
  url "https://github.com/ashishodu2023/TideVec/archive/refs/tags/v0.3.0.tar.gz"
  sha256 "0000000000000000000000000000000000000000000000000000000000000000"
  license "Apache-2.0"
  head "https://github.com/ashishodu2023/TideVec.git", branch: "main"

  livecheck do
    url :stable
    strategy :github_latest
  end

  depends_on "cmake" => :build
  depends_on "ninja" => :build

  on_macos do
    depends_on xcode: ["15.0", :build] # C++20
  end

  fails_with :gcc do
    version "11"
    cause "Requires C++20 library features from GCC 12+"
  end

  def install
    args = %W[
      -DTIDEVEC_BUILD_TESTS=OFF
      -DCMAKE_CXX_STANDARD=20
    ]
    # The default is -march=native; bottles must run on any supported CPU,
    # so let Homebrew's compiler shims choose the target instead.
    args << "-DTIDEVEC_MARCH=" if Hardware::CPU.intel?

    system "cmake", "-S", ".", "-B", "build", "-G", "Ninja", *args, *std_cmake_args
    system "cmake", "--build", "build", "--target", "tidevec-server"
    bin.install "build/tidevec-server"
  end

  def post_install
    (var/"tidevec/data").mkpath
    (var/"tidevec/backups").mkpath
  end

  def caveats
    <<~EOS
      The service listens on 127.0.0.1:6399 with authentication disabled and keeps
      its data in #{var}/tidevec. To require an API key, run the server yourself:
        TIDEVEC_API_KEY=... tidevec-server --data-dir #{var}/tidevec/data --require-auth
      Python client: pip install tidevec   ·   TypeScript: npm install tidevec
    EOS
  end

  service do
    run [opt_bin/"tidevec-server",
         "--host", "127.0.0.1",
         "--port", "6399",
         "--data-dir", var/"tidevec/data",
         "--backup-dir", var/"tidevec/backups",
         "--no-require-auth",
         "--quiet"]
    keep_alive true
    working_dir var/"tidevec"
    log_path var/"log/tidevec.log"
    error_log_path var/"log/tidevec.log"
  end

  test do
    port = free_port
    data = testpath/"data"
    pid = spawn bin/"tidevec-server", "--host", "127.0.0.1", "--port", port.to_s,
                "--data-dir", data.to_s, "--backup-dir", (testpath/"backups").to_s,
                "--no-require-auth", "--quiet"
    begin
      health = nil
      30.times do
        sleep 0.5
        health = shell_output("curl -fsS http://127.0.0.1:#{port}/health 2>/dev/null || true")
        break if health.include?("\"status\"")
      end
      assert_match(/"version":\s*"#{Regexp.escape(version.to_s)}"/, health)

      api = "http://127.0.0.1:#{port}/v1/collections"
      json = "Content-Type: application/json"
      system "curl", "-fsS", "-H", json, "-d", '{"name":"t","dim":3}', api
      system "curl", "-fsS", "-H", json,
             "-d", '{"vectors":[{"id":"a","embedding":[1,0,0]}]}', "#{api}/t/upsert"
      out = shell_output("curl -fsS -H '#{json}' -d '{\"vector\":[1,0,0],\"top_k\":1}' #{api}/t/search")
      assert_match(/"id":\s*"a"/, out)
    ensure
      Process.kill("TERM", pid)
      Process.wait(pid)
    end
  end
end
