(ns raster.gpu.device-probe
  "Structured test gates for optional native runtimes and explicit qualification lanes.

   A missing device is a legitimate, visible skip on generic CI.  A namespace load failure is a
   test failure.  Driver/query failures are visible skips unless RASTER_EXPECT_OPENCL declares
   that this lane must execute OpenCL.  Capability decisions always inspect the exact device bound
   by the singleton runtime, never some other enumerated device."
  (:require [clojure.string :as str]
            [clojure.test :refer [is]]))

(defn- truthy-environment?
  [name]
  (contains? #{"1" "true" "yes" "on"}
             (some-> (System/getenv name) str/lower-case)))

(def ^:dynamic *expect-opencl?*
  (truthy-environment? "RASTER_EXPECT_OPENCL"))

(defn probe-opencl-with
  "Classify an OpenCL runtime using injectable operations.  Public for the gate's unit tests."
  [{:keys [load! query-devices init! selected-device-info]}]
  (let [loaded (try
                 (load!)
                 {:ok true}
                 (catch Throwable error
                   {:ok false :error error}))]
    (if-not (:ok loaded)
      {:backend :opencl :status :load-failed :error (:error loaded)}
      (try
        (let [devices (query-devices)]
          (if-not (seq devices)
            {:backend :opencl :status :no-device}
            (do
              (init!)
              (let [device (selected-device-info)]
                (when-not (map? device)
                  (throw (ex-info "OpenCL runtime did not identify its selected device"
                                  {:selected device})))
                {:backend :opencl
                 :status :available
                 :n-devices (count devices)
                 :device device}))))
        (catch Throwable error
          {:backend :opencl :status :probe-error :error error})))))

(defn- runtime-api
  []
  (require 'raster.gpu.ocl-runtime)
  (let [api {:load! (constantly true)
             :query-devices (resolve 'raster.gpu.ocl-runtime/query-devices)
             :init! (resolve 'raster.gpu.ocl-runtime/init!)
             :selected-device-info (resolve 'raster.gpu.ocl-runtime/selected-device-info)}]
    (doseq [[operation f] (dissoc api :load!)]
      (when-not (ifn? f)
        (throw (ex-info "OpenCL runtime API is incomplete" {:operation operation}))))
    api))

(def opencl-status
  (delay
    (let [loaded (try
                   {:ok true :api (runtime-api)}
                   (catch Throwable error
                     {:ok false :error error}))]
      (if (:ok loaded)
        (probe-opencl-with (:api loaded))
        {:backend :opencl :status :load-failed :error (:error loaded)}))))

(defn- extension-set
  [device]
  (into #{} (remove str/blank?) (str/split (or (:extensions device) "") #"\s+")))

(defn capability-supported?
  [device capability]
  (case capability
    :fp16 (contains? (extension-set device) "cl_khr_fp16")
    :fp64 (let [extensions (extension-set device)]
            (or (contains? extensions "cl_khr_fp64")
                (contains? extensions "cl_amd_fp64")))
    ;; Tuned dispatch alternatives (subgroup score reuse, matrix leaves) are emitted for GPU
    ;; descriptors only; a CPU OpenCL device such as PoCL executes the portable kernels.
    :gpu-device (= :gpu (:type device))
    ;; Indexed attention and the scheduled reductions use sub-group lanes; an OpenCL 1.2 CPU
    ;; implementation without cl_khr_subgroups (PoCL 1.8 on CI images) cannot build them.
    :intel-subgroups (contains? (extension-set device) "cl_intel_subgroups")
    :subgroups (let [extensions (extension-set device)]
                 (or (contains? extensions "cl_khr_subgroups")
                     (contains? extensions "cl_intel_subgroups")))
    (throw (ex-info "Unknown OpenCL test capability" {:capability capability}))))

(defn opencl-status-for
  [capability]
  (let [status @opencl-status]
    (if (and (= :available (:status status))
             capability
             (not (capability-supported? (:device status) capability)))
      (assoc status :status :missing-capability :capability capability)
      status)))

(def opencl-available?
  (delay (= :available (:status @opencl-status))))

(def opencl-fp16-available?
  (delay (= :available (:status (opencl-status-for :fp16)))))

(def opencl-fp64-available?
  (delay (= :available (:status (opencl-status-for :fp64)))))

(def opencl-gpu-available?
  (delay (= :available (:status (opencl-status-for :gpu-device)))))

(def opencl-subgroups-available?
  (delay (= :available (:status (opencl-status-for :subgroups)))))

(defonce ^:private opencl-skip-log (atom {}))

(defonce ^:private opencl-summary-hook
  (delay
    (.addShutdownHook
     (Runtime/getRuntime)
     (Thread.
      (fn []
        (let [status @opencl-status
              skips @opencl-skip-log
              total (reduce + 0 (vals skips))]
          (println
           (format "  [OPENCL SUITE] probe=%s%s | %d test(s) took the skip path%s"
                   (name (:status status))
                   (if-let [device (get-in status [:device :name])]
                     (str " (" device ")")
                     "")
                   total
                   (if (pos? total) (str " " (pr-str skips)) "")))))))
    true))

(defn opencl-skip!
  "Record one visible OpenCL skip marker, or fail when the runtime is broken/required.

   `capability` is nil, :fp16, :fp64, :gpu-device or :subgroups. Missing optional capabilities remain honest skips;
   RASTER_EXPECT_OPENCL requires a usable device but does not imply every optional extension."
  ([test-label] (opencl-skip! test-label nil))
  ([test-label capability]
   @opencl-summary-hook
   (let [{:keys [status error device] :as result} (opencl-status-for capability)]
     (swap! opencl-skip-log update (or capability status) (fnil inc 0))
     (case status
       :load-failed
       (is false
           (str "[OPENCL LOAD FAILED] " test-label " — " (some-> error .getMessage)
                ". This is a runtime breakage, not an absent device."))

       :probe-error
       (if *expect-opencl?*
         (is false
             (str "[OPENCL PROBE FAILED] " test-label " — " (some-> error .getMessage)
                  ". RASTER_EXPECT_OPENCL requires this lane to execute OpenCL."))
         (do
           (println (str "  [OPENCL SKIP/WARN] " test-label " — "
                         (some-> error .getMessage)))
           (is true "opencl-skip-marker")))

       :no-device
       (if *expect-opencl?*
         (is false
             (str "[OPENCL DEVICE MISSING] " test-label
                  " — RASTER_EXPECT_OPENCL requires an OpenCL device."))
         (do
           (println (str "  [OPENCL SKIP] " test-label " — no requested OpenCL device"))
           (is true "opencl-skip-marker")))

       :missing-capability
       (do
         (println (str "  [OPENCL SKIP] " test-label " — selected device " (:name device)
                       " lacks " (name capability)))
         (is true "opencl-skip-marker"))

       :available
       (throw
        (IllegalStateException.
         (str "opencl-skip! called for " test-label " while OpenCL is available")))

       (throw (ex-info "Unknown OpenCL probe status" {:result result}))))))


(defn probe-cuda-with
  "Classify the context-free driver query without conflating a namespace failure with absence."
  [{:keys [load! probe!]}]
  (let [loaded (try (load!) {:ok true}
                    (catch Throwable error {:ok false :error error}))]
    (if-not (:ok loaded)
      {:backend :cuda :status :load-failed :error (:error loaded)}
      (try
        (let [result (probe!)
              status (:status result)]
          (case status
            :ready (if (seq (:devices result))
                     (assoc result :backend :cuda :status :available)
                     (throw (ex-info "ready CUDA probe has no device" {:result result})))
            (:no-device :unavailable) (assoc result :backend :cuda)
            :driver-error (assoc result :backend :cuda :status :probe-error
                                 :error (ex-info "CUDA driver initialization failed" result))
            (throw (ex-info "unknown CUDA driver probe status" {:result result}))))
        (catch Throwable error {:backend :cuda :status :probe-error :error error})))))

(def cuda-status
  (delay
    (probe-cuda-with
     {:load! #(require 'raster.gpu.cuda-driver-probe)
      :probe! #(let [probe (resolve 'raster.gpu.cuda-driver-probe/probe)]
                 (when-not (ifn? probe)
                   (throw (ex-info "CUDA driver probe API is incomplete" {})))
                 (probe))})))

(def cuda-available? (delay (= :available (:status @cuda-status))))

(defonce ^:private native-skip-log (atom {}))

(defonce ^:private native-summary-hook
  (delay
    (.addShutdownHook
     (Runtime/getRuntime)
     (Thread.
      (fn []
        ;; Explicitly unrequested tests must not start a driver probe at JVM shutdown.
        (println
         (str "  [NATIVE SUITE] recorded gate markers " (pr-str @native-skip-log))))))
    true))

(defn native-skip!
  "Report one classified native gate. Unrequested is distinct from unavailable hardware;
   required lanes and broken namespace loads fail. Available/unknown states cannot skip."
  [{:keys [backend status error] :as result} label required?]
  (when-not (contains? #{:cuda :nvrtc} backend)
    (throw (ex-info "unknown native test backend" {:result result})))
  (when-not (contains? #{:not-requested :no-device :unavailable :probe-error
                        :missing-toolchain :load-failed} status)
    (throw (ex-info "native skip needs an actual classified unavailable gate" {:result result})))
  @native-summary-hook
  (swap! native-skip-log update [backend status] (fnil inc 0))
  (let [message (str "[" (name backend) " " (name status) "] " label
                     (when error (str " — " (.getMessage error))))]
    (if (or required? (= :load-failed status))
      (is false (str message ". This lane must execute; no skip is permitted."))
      (do
        (println (str "  [NATIVE SKIP] " message))
        (is true (str "native-skip-marker: " message))))))

(defn cuda-skip!
  ([label]
   (native-skip! @cuda-status label (= "1" (System/getenv "RASTER_REQUIRE_CUDA_DEVICE"))))
  ([label mode]
   (when-not (= :not-requested mode)
     (throw (ex-info "unknown CUDA qualification gate mode" {:mode mode})))
   (native-skip! {:backend :cuda :status :not-requested} label
                 (= "1" (System/getenv "RASTER_REQUIRE_CUDA_DEVICE")))))

(defn nvrtc-skip!
  [label]
  (native-skip! {:backend :nvrtc :status :missing-toolchain} label
                (= "1" (System/getenv "RASTER_REQUIRE_NVRTC"))))
