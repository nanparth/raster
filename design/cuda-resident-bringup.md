# CUDA resident bring-up

Raster already emits generated CUDA C++ from verified KernelBody and checks the resulting
sm_80 PTX with `nvcc` in hardware-free CI. Source legality remains separate from execution.
The initial CUDA Driver API adapter now admits `:cuda:0` through the shared resident backend
registry, with explicit device memory and borrowed pointer views. Other CUDA ordinals and HIP
remain unsupported resident routes. Discovery uses the read-only driver probe; creating a
session rejects a conflicting already-selected CUDA ordinal rather than silently using it.

The initial device tests exercise generated maps, ordered arguments, offset views, changed-input
replay, events and the public Compiled/LinkPlan path. They do not establish full runtime parity:
asynchronous range transfers, device copies and broader numerical routes remain subsequent
gates. Capability queries report asynchronous transfer support as false, and unsupported
operations raise explicit backend/operation errors rather than falling back or synchronizing.
Native CUDA Graph capture is not implied by logical graph replay.

## First device gate

Use one public equation-first elementwise map from the existing compile-fixture corpus.
The path must be `deftm → typed SOAC → schedule → KernelBody → CUDA source → module`, then:

1. Probe the installed CUDA driver and enumerate a real device, recording device name,
   compute capability, driver version, toolkit/compiler version, and available memory.
   `raster.gpu.cuda-driver-probe/probe` now performs the read-only driver/name/capability
   part without creating a context. Its status is a current observation, not a permanent
   hardware claim. The resident adapter queries memory and launch limits. CUDA driver API
   compatibility version is named separately from the installed NVIDIA driver build.
   On a CUDA host, run its test with `RASTER_REQUIRE_CUDA_DEVICE=1` so a missing device fails.
2. Create a resident session; allocate input/output buffers; upload the input through the
   existing typed buffer contract. Compile the generated source, load its module, bind its
   **ordered KernelABI**, and launch its **KernelLaunch** 1–3D geometry. No fixture-specific
   binder or hand-written kernel may be substituted.
3. Complete the launch through the backend-neutral event contract; download output and
   compare against the uncompiled JVM result. Exercise an uneven extent and a nonzero
   BufferView offset so a correct-looking zero-origin launch is not enough.
4. Replay the bound call with changed resident input, then close the session and verify
   resource cleanup. A device event supplies the elapsed-kernel measurement; host timing
   is recorded separately and is not used to assert a speedup.

The initial adapter should use the CUDA Driver API for context, memory, module, launch and
event operations. The hardware-free `raster.gpu.cuda-nvrtc` transport now compiles generated
CUDA source to PTX in-process; `cuda-driver` and `cuda-runtime` load and launch that PTX
through the Driver API, without an `nvcc` subprocess on each compilation. Keep
the existing `nvcc` compile gate. Record compilation identity and driver/toolkit compatibility
in the artifact cache key. A precompiled PTX fixture is acceptable for the *first* transport
smoke, but the gate above is not complete until Raster compiles its generated source itself.
NVRTC requires explicit toolkit include paths and uses small virtual C headers for the
`stdint.h`/`math.h` names in the current CUDA preamble; host glibc headers are not a valid
substitute. Runtime initialization accepts explicit `:include-paths`. For ordinary sessions,
`RASTER_CUDA_INCLUDE_PATHS` supplies paths separated by the platform path separator; otherwise
an explicitly configured `CUDA_HOME` supplies its `include` directory. Spaces within paths
are preserved. No toolkit installation path is committed in source.
An NVRTC-equipped host should run the optional transport test with
`RASTER_REQUIRE_NVRTC=1` so missing native coverage is a failure, not a skip.

Do not mark `:cuda` a resident backend merely because a library or `nvcc` is present.
The backend entry must declare its memory space, coherence, slice ownership, transfer
capabilities, event behavior and physical-queue facts. Unsupported graph or async operations
should fail explicitly; they must not silently become synchronous or fall back to OpenCL.

`raster.gpu.core` currently resolves many historical runtime functions, including
`invoke-registered-map-void-kernel` and `bind-registered-map-void-kernel`. These are **not**
requirements for a new CUDA backend. The first gate uses the modern `KernelArtifact`,
`KernelCall` and ordered ABI path: arena creation/close, resident buffer allocation/free,
transfer, artifact registration, call binding, launch/completion and download. Add graph
record/replay and asynchronous range batches at the subsequent gate. Where the public
session layer still calls a legacy operation, report an unsupported-route diagnostic
with its operation and backend; do not implement a second CUDA source compiler to satisfy it.

## Subsequent gates

1. Ordered multi-kernel `KernelGraph` recording/replay with stable resident bindings,
   ranged views, explicit completion and cancellation-safe resource retention.
2. A generated reduction (RMSNorm) and generated Q4_K projection, each checked against
   its independent numerical oracle. The Q4_K integer component requires bit identity;
   floating reduction association is reported rather than assumed.
3. Generated GEMM and routed attention, with matching layout/precision/shape against
   NVIDIA library and framework baselines. Device-event timings, compile time, memory
   traffic and numerical error are reported separately. Tune only after correctness.
4. Expand device capabilities and selected schedule families from observed hardware;
   direct PTX remains a later target dialect if CUDA C++ plus vendor compilation is shown
   to leave important hardware behavior inaccessible.

The CUDA host need not be part of normal CI. CI continues to compile generated source
without a GPU; a short opt-in device suite runs on the CUDA host. Avoid renting one until
the first-gate smoke command and driver adapter are ready. At that point, access to a Linux
host with an NVIDIA GPU, working driver, CUDA toolkit/NVRTC and remote shell is enough;
record the exact versions rather than assuming an architecture in the compiler.

## Running the initial resident proof

On a configured CUDA host, require actual execution rather than accepting optional skips:

```sh
RASTER_REQUIRE_CUDA_DEVICE=1 RASTER_REQUIRE_NVRTC=1 clojure -M:test \
  -n raster.gpu.cuda-runtime-device-test \
  -n raster.gpu.resident-device-admission-test
```

Record the exact source revision/diff, toolkit/header options, target architecture, live device
identity, outcome and skips. Run the same generated paths under Compute Sanitizer with explicit
memory leak checking. Event timings in a correctness smoke are measurement mechanics, not
performance evidence. Preserve independent arithmetic references, existing numerical thresholds
and matrix-specific alignment/stride checks before claiming wider operation support.
