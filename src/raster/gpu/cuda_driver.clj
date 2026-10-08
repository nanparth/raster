(ns raster.gpu.cuda-driver
  "CUDA Driver API transport. Runtime ownership belongs to resource-cleanup/root;
   this leaf marshals native calls and brackets the calling thread's context stack."
  (:import [java.lang.foreign Arena FunctionDescriptor Linker Linker$Option
            MemoryLayout MemorySegment SymbolLookup ValueLayout]))

(def I32 ValueLayout/JAVA_INT)
(def I64 ValueLayout/JAVA_LONG)
(def PTR ValueLayout/ADDRESS)
(def ^:dynamic *invoke* nil)
(def ^:dynamic *context* nil)

(def ^:private library
  (delay
    (or (some (fn [name]
                (try (SymbolLookup/libraryLookup name (Arena/global))
                     (catch Exception _ nil))) ["libcuda.so.1" "libcuda.so"])
        (throw (ex-info "CUDA driver is unavailable" {:reason :cuda-driver-unavailable})))))
(def ^:private handles (atom {}))

(defn- raw! [symbol layouts arguments]
  (let [key [symbol layouts]
        handle (or (get @handles key)
                   (let [found (.find @library symbol)]
                     (when-not (.isPresent found)
                       (throw (ex-info "CUDA driver symbol is unavailable"
                                       {:reason :cuda-driver-symbol-unavailable :symbol symbol})))
                     (let [h (.downcallHandle (Linker/nativeLinker) (.get found)
                                             (FunctionDescriptor/of I32 (into-array MemoryLayout layouts))
                                             (into-array Linker$Option []))]
                       (swap! handles assoc key h) h)))
        result (int (.invokeWithArguments handle (into-array Object arguments)))]
    (when-not (zero? result)
      (throw (ex-info "CUDA driver operation failed"
                      {:reason :cuda-driver-error :operation symbol :error-code result})))
    nil))

(defn- output! [arena layout symbol layouts args]
  (let [out (.allocate arena layout)]
    (raw! symbol (into [PTR] layouts) (into [out] args))
    (.get out layout (long 0))))

(defn- in-context [context f]
  (if-not context
    (f)
    (do
      (raw! "cuCtxPushCurrent_v2" [PTR] [context])
      (let [failure (volatile! nil)]
        (try (f)
             (catch Throwable e (vreset! failure e) (throw e))
             (finally
               (try
                 (with-open [arena (Arena/ofConfined)]
                   (let [popped (output! arena PTR "cuCtxPopCurrent_v2" [] [])]
                     (when-not (= (.address context) (.address popped))
                       (throw (ex-info "CUDA context stack changed during operation"
                                       {:reason :cuda-context-stack-mismatch})))))
                 (catch Throwable e
                   (if-let [primary @failure] (.addSuppressed primary e) (throw e))))))))))

(defn- initialize! [ordinal]
  (with-open [arena (Arena/ofConfined)]
    (raw! "cuInit" [I32] [(int 0)])
    (let [device (output! arena I32 "cuDeviceGet" [I32] [(int ordinal)])
          context (output! arena PTR "cuDevicePrimaryCtxRetain" [I32] [device])]
      (try
        (in-context context
          (fn []
            (let [stream (output! arena PTR "cuStreamCreate" [I32] [(int 1)])]
              (try
                (let [major (.allocate arena I32) minor (.allocate arena I32)
                      name (.allocate arena (long 256))
                      version (output! arena I32 "cuDriverGetVersion" [] [])
                      free (.allocate arena I64) total (.allocate arena I64)
                      attribute (fn [id] (output! arena I32 "cuDeviceGetAttribute" [I32 I32] [(int id) device]))]
                  (raw! "cuDeviceComputeCapability" [PTR PTR I32] [major minor device])
                  (raw! "cuDeviceGetName" [PTR I32 I32] [name (int 256) device])
                  (raw! "cuMemGetInfo_v2" [PTR PTR] [free total])
                  {:context context :stream stream :device device
                   :name (.getString name (long 0)) :driver-api-version version
                   :max-workgroup-size (attribute 1)
                   :max-block-dimensions (mapv attribute [2 3 4])
                   :max-grid-dimensions (mapv attribute [5 6 7])
                   :shared-local-memory (attribute 8) :warp-size (attribute 10)
                   :multiprocessor-count (attribute 16)
                   :compute-capability [(.get major I32 (long 0)) (.get minor I32 (long 0))]
                   :free-memory-bytes (.get free I64 (long 0))
                   :total-memory-bytes (.get total I64 (long 0))})
                (catch Throwable e
                  (try (raw! "cuStreamDestroy_v2" [PTR] [stream])
                       (catch Throwable rollback (.addSuppressed e rollback)))
                  (throw e))))))
        (catch Throwable e
          (try (raw! "cuDevicePrimaryCtxRelease_v2" [I32] [device])
               (catch Throwable rollback (.addSuppressed e rollback)))
          (throw e))))))

(defn- native-invoke [operation args]
  (with-open [arena (Arena/ofConfined)]
    (case operation
      :initialize (initialize! (first args))
      :destroy-runtime (let [{:keys [context stream device]} (first args)]
                         (in-context context #(raw! "cuStreamDestroy_v2" [PTR] [stream]))
                         (raw! "cuDevicePrimaryCtxRelease_v2" [I32] [device]))
      :allocate (output! arena I64 "cuMemAlloc_v2" [I64] [(long (first args))])
      :free (raw! "cuMemFree_v2" [I64] [(long (first args))])
      :module-load (let [bytes (first args) data (.allocate arena (long (alength bytes)))]
                     (MemorySegment/copy (MemorySegment/ofArray bytes) (long 0) data (long 0) (long (alength bytes)))
                     (output! arena PTR "cuModuleLoadData" [PTR] [data]))
      :module-function (let [[module name] args]
                         (output! arena PTR "cuModuleGetFunction" [PTR PTR]
                                  [module (.allocateFrom arena (str name))]))
      :module-unload (raw! "cuModuleUnload" [PTR] args)
      :launch (let [[function geometry parameters stream] args
                    wg (take 3 (concat (:workgroup-size geometry) [1 1 1]))
                    gc (take 3 (concat (:group-count geometry) [1 1 1]))]
                (raw! "cuLaunchKernel" [PTR I32 I32 I32 I32 I32 I32 I32 PTR PTR PTR]
                      (into [function] (concat (map int gc) (map int wg)
                                              [(int (:shared-memory-bytes geometry)) stream parameters MemorySegment/NULL]))))
      :stream-sync (raw! "cuStreamSynchronize" [PTR] args)
      :upload (let [[pointer segment bytes] args]
                (raw! "cuMemcpyHtoD_v2" [I64 PTR I64] [(long pointer) segment (long bytes)]))
      :download (let [[pointer segment bytes] args]
                  (raw! "cuMemcpyDtoH_v2" [PTR I64 I64] [segment (long pointer) (long bytes)]))
      :copy (let [[dst src bytes stream] args]
              (raw! "cuMemcpyDtoDAsync_v2" [I64 I64 I64 PTR] [(long dst) (long src) (long bytes) stream]))
      :event-create (output! arena PTR "cuEventCreate" [I32] [(int 0)])
      :event-record (raw! "cuEventRecord" [PTR PTR] args)
      :event-sync (raw! "cuEventSynchronize" [PTR] args)
      :event-destroy (raw! "cuEventDestroy_v2" [PTR] args)
      :event-query (try (raw! "cuEventQuery" [PTR] args) true
                        (catch clojure.lang.ExceptionInfo e
                          (if (= 600 (:error-code (ex-data e))) false (throw e))))
      :event-elapsed (let [[start end] args
                           out (.allocate arena ValueLayout/JAVA_FLOAT)]
                       (raw! "cuEventElapsedTime" [PTR PTR PTR] [out start end])
                       (* 1000000.0 (double (.get out ValueLayout/JAVA_FLOAT (long 0)))))
      (throw (ex-info "Unsupported CUDA transport operation"
                      {:reason :cuda-unsupported-operation :backend :cuda :operation operation})))))

(defn invoke!
  "Invoke one driver operation under the exact runtime context on this thread.
   The dynamic seam is for contract/fault tests; production always uses the Driver API."
  [operation arguments]
  (if *invoke*
    (*invoke* operation arguments)
    (in-context (when-not (contains? #{:initialize :destroy-runtime} operation) *context*)
                #(native-invoke operation arguments))))
