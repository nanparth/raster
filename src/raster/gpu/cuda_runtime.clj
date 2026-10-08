(ns raster.gpu.cuda-runtime
  "Modern ordered KernelCall transport over explicit CUDA device memory.
   The shared resident registry admits the limited public CUDA ordinal-zero backend;
   compiler targets remain broader than the operations implemented by this adapter."
  (:refer-clojure :exclude [reset!])
  (:require [clojure.string :as string]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.ir.buffer-view :as view]
            [raster.compiler.ir.kernel-abi :as abi]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-call :as call]
            [raster.gpu.cuda-driver :as driver]
            [raster.gpu.cuda-execution :as execution]
            [raster.gpu.cuda-nvrtc :as nvrtc]
            [raster.gpu.resource-cleanup :as cleanup]
            [raster.gpu.runtime-root :as root])
  (:import [java.lang.foreign Arena MemorySegment ValueLayout]))

(def state (atom {}))
(def kernel-registry (atom {}))
(def ^:dynamic *current-arena* nil)

(defn include-paths-from-env
  "Pure compiler configuration policy. Explicit include-path options override this default."
  [env]
  (if (contains? env "RASTER_CUDA_INCLUDE_PATHS")
    (if (seq (get env "RASTER_CUDA_INCLUDE_PATHS"))
      (vec (string/split (get env "RASTER_CUDA_INCLUDE_PATHS")
                         (re-pattern (java.util.regex.Pattern/quote java.io.File/pathSeparator))))
      [])
    (if (seq (get env "CUDA_HOME")) [(str (get env "CUDA_HOME") "/include")] [])))

(defn- native! [operation arguments]
  (binding [driver/*context* (:context @state)]
    (driver/invoke! operation arguments)))

(defn initialize!
  "Initialize the shared process root. Toolkit include paths are explicit configuration,
   never inferred from a compiler device descriptor or committed machine paths."
  ([] (initialize! {}))
  ([opts]
   (let [device-ordinal (get opts :device-ordinal 0)
         include-paths (get opts :include-paths (include-paths-from-env (System/getenv)))]
   (when-not (and (integer? device-ordinal) (<= 0 device-ordinal Integer/MAX_VALUE)
                  (vector? include-paths) (every? string? include-paths))
     (throw (ex-info "Invalid CUDA runtime configuration" {:reason :cuda-runtime-configuration})))
   (locking state
     (when (and (:initialized? @state)
                (not= [device-ordinal include-paths] [(:device-ordinal @state) (:include-paths @state)]))
       (throw (ex-info "CUDA runtime already has a different device/compiler configuration"
                       {:reason :cuda-runtime-configuration-mismatch})))
     (root/initialize!
      state [{:id :runtime :release #(binding [driver/*context* (:context %)]
                                     (driver/invoke! :destroy-runtime [%]))}]
      (fn [entry]
        (assoc (root/acquire! entry :runtime #(driver/invoke! :initialize [device-ordinal]))
               :include-paths include-paths :device-ordinal device-ordinal))))
   nil)))

(defn ensure-init! [] (when-not (:initialized? @state) (initialize!)) (root/assert-live! state))

(defn execution-device-info []
  (ensure-init!)
  (select-keys @state [:name :device-ordinal :compute-capability :driver-api-version
                      :max-workgroup-size :max-block-dimensions :max-grid-dimensions
                      :warp-size :shared-local-memory :multiprocessor-count
                      :free-memory-bytes :total-memory-bytes]))

(defn- owner! [value]
  (or (cleanup/lifetime-owner value)
      (throw (ex-info "CUDA value has no lifetime owner" {:reason :missing-cleanup-owner}))))

(defn device-buffer? [value] (true? (::buffer value)))

(defn assert-buffer-live! [buffer]
  (when-not (device-buffer? buffer)
    (throw (ex-info "CUDA requires a resident buffer" {:reason :cuda-buffer-type})))
  (cleanup/assert-live! (owner! buffer))
  (root/assert-live! state)
  (when-not (identical? (::context buffer) (:context @state))
    (throw (ex-info "CUDA buffer belongs to another context generation"
                    {:reason :runtime-generation-mismatch})))
  buffer)

(defn- assert-unborrowed! [borrows reason message]
  (when (seq @borrows)
    (throw (ex-info message {:reason reason :cleanup-retry-safe? true}))))

(defn make-buffer
  ([n] (make-buffer n :float))
  ([n dt] (make-buffer n dt {}))
  ([n dt {:keys [retain-owner! adopt-cleanup!]}]
   (when-not (dtype/known? dt)
     (throw (ex-info "CUDA buffer dtype is unsupported"
                     {:reason :cuda-unsupported-operation :backend :cuda :operation :make-buffer :dtype dt})))
   (when-not (and (integer? n) (<= 0 n Long/MAX_VALUE))
     (throw (ex-info "GPU buffer element count must be non-negative"
                     {:reason :gpu-buffer-negative-elements :elements n})))
   (when-not (every? #(or (nil? %) (fn? %)) [retain-owner! adopt-cleanup!])
     (throw (ex-info "Buffer ownership requires callbacks" {:reason :invalid-cleanup-plan})))
   (let [dt (dtype/canon dt) bytes (Math/multiplyExact (long n) (long (dtype/bytes-of dt)))
         slot (cleanup/acquisition-slot) borrows (atom #{})]
     (ensure-init!)
     (root/construct-child!
      state [{:id :memory
              :release #(do
                          (assert-unborrowed! borrows :cuda-buffer-in-use
                                              "CUDA buffer is retained by a prepared call")
                          (cleanup/release-native! slot (fn [p] (native! :free [p]))))}]
      (fn [owner entry]
        (when retain-owner! (retain-owner! owner))
        (let [p (cleanup/acquire-native! slot #(native! :allocate [(max 1 bytes)]))
              allocation (view/allocation {:id (random-uuid) :byte-size bytes
                                           :memory-space :device :device :cuda:0
                                           :alignment 256 :coherence :explicit-transfer})]
          {::buffer true ::context (:context @(:projection entry))
           :device-pointer p :n-elements n :byte-size bytes :dtype dt :alignment 256
           :borrows borrows
           :view (view/view allocation {:id (random-uuid) :dtype dt :shape [n]})}))
      (or adopt-cleanup! retain-owner!)))))

(defn make-buffer-like [buffer] (make-buffer (:n-elements buffer) (:dtype buffer)))

(defn slice-buffer [buffer byte-offset byte-length dt]
  (assert-buffer-live! buffer)
  (let [dt (dtype/canon dt) width (dtype/bytes-of dt)]
    (when-not (= dt (:dtype buffer))
      (throw (ex-info "CUDA slice cannot reinterpret its parent dtype" {:reason :cuda-slice-dtype})))
    (when-not (and (integer? byte-offset) (integer? byte-length)
                   (<= 0 byte-offset) (<= 0 byte-length)
                   (<= (+' byte-offset byte-length) (:byte-size buffer))
                   (zero? (mod byte-offset width)) (zero? (mod byte-length width)))
      (throw (ex-info "CUDA slice range is invalid" {:reason :cuda-slice-range})))
    (-> buffer
        (dissoc ::cleanup/owner)
        (assoc ::cleanup/lifetime-owner (owner! buffer)
               :device-pointer (Math/addExact (long (:device-pointer buffer)) (long byte-offset))
               ;; BufferAllocation guarantees are powers of two. Their gcd with an offset is
               ;; the smaller of the parent's guarantee and the offset's lowest set bit.
               :alignment (if (zero? byte-offset) (:alignment buffer)
                              (min (:alignment buffer) (Long/lowestOneBit (long byte-offset))))
               :byte-size byte-length :n-elements (quot byte-length width)
               :view (view/subview (:view buffer) {:id (random-uuid) :byte-offset byte-offset
                                                 :shape [(quot byte-length width)]})))))

(defn free-buffer! [buffer]
  (when-not (::cleanup/owner buffer)
    (throw (ex-info "Borrowed CUDA slice has no destruction authority" {:reason :missing-cleanup-owner})))
  (assert-unborrowed! (:borrows buffer) :cuda-buffer-in-use "CUDA buffer is retained by a prepared call")
  (cleanup/release! (::cleanup/owner buffer)))

(defn- host-segment [host]
  (if (instance? MemorySegment host) host (MemorySegment/ofArray host)))

(defn plan-range [buffer host {:keys [src-element dst-element elements]
                               :or {src-element 0 dst-element 0}} direction]
  (assert-buffer-live! buffer)
  (when-not (contains? #{:upload :download} direction)
    (throw (ex-info "CUDA range direction is unsupported" {:reason :cuda-range-direction})))
  (let [segment (host-segment host) width (dtype/bytes-of (:dtype buffer))
        [buf-index host-index] (if (= :upload direction) [dst-element src-element] [src-element dst-element])]
    (when-not (and (every? #(and (integer? %) (<= 0 % Long/MAX_VALUE)) [buf-index host-index elements])
                   (<= (+' buf-index elements) (:n-elements buffer))
                   (<= (*' (+' host-index elements) width) (.byteSize segment)))
      (throw (ex-info "CUDA transfer range exceeds buffer/host capacity" {:reason :cuda-range-capacity})))
    {:buf-off (* buf-index width) :host-off (* host-index width)
     :n-bytes (* elements width) :host-seg segment}))

(defn execute-range! [buffer {:keys [buf-off host-off n-bytes host-seg]} direction]
  (assert-buffer-live! buffer)
  (when (pos? n-bytes)
    (with-open [arena (Arena/ofConfined)]
      (let [staging (.allocate arena (long n-bytes))]
        ;; Synchronous host copies also order against previous launches on this stream.
        (native! :stream-sync [(:stream @state)])
        (when (= :upload direction)
          (MemorySegment/copy host-seg (long host-off) staging (long 0) (long n-bytes)))
        (native! direction [(+' (:device-pointer buffer) buf-off) staging n-bytes])
        (when (= :download direction)
          (MemorySegment/copy staging (long 0) host-seg (long host-off) (long n-bytes))))))
  buffer)

(defn upload-range! [buffer host opts] (execute-range! buffer (plan-range buffer host opts :upload) :upload))
(defn download-range! [buffer host opts]
  (execute-range! buffer (plan-range buffer host opts :download) :download)
  host)
(defn array->buffer! [buffer host]
  (upload-range! buffer host {:elements (:n-elements buffer)}))
(defn buffer-of-array [host]
  (let [dt (case (.getName (.getComponentType (class host)))
             "float" :float "double" :double "int" :int "long" :long "short" :half "byte" :byte)
        buffer (make-buffer (alength host) dt)]
    (try (array->buffer! buffer host)
         (catch Throwable e (try (free-buffer! buffer) (catch Throwable x (.addSuppressed e x))) (throw e)))))
(defn buffer->array [buffer]
  (let [host (case (:dtype buffer)
               :float (float-array (:n-elements buffer)) :double (double-array (:n-elements buffer))
               :int (int-array (:n-elements buffer)) :long (long-array (:n-elements buffer))
               :half (short-array (:n-elements buffer)) :byte (byte-array (:n-elements buffer)))]
    (download-range! buffer host {:elements (:n-elements buffer)}) host))

(defn expand-pointer-binding [{:keys [slots binding]} value]
  (when-not (= 1 (count slots))
    (throw (ex-info "CUDA composite pointer binding is unsupported"
                    {:reason :cuda-unsupported-operation :backend :cuda :operation :expand-pointer-binding
                     :binding binding})))
  [(assert-buffer-live! value)])

(defn make-kernel-arena! []
  (ensure-init!)
  (when-not (zero? (:device-ordinal @state))
    (throw (ex-info "Public CUDA sessions require ordinal zero"
                    {:reason :cuda-runtime-configuration-mismatch :requested-device-ordinal 0
                     :active-device-ordinal (:device-ordinal @state)})))
  (keyword (str "cuda-arena-" (random-uuid))))

(defn register-kernel!
  ([name kernel] (register-kernel! name kernel *current-arena*))
  ([name kernel arena]
   (artifact/validate! kernel)
   (when-not (and (= name (:kernel-name kernel)) (= :cuda-c (:target kernel)))
     (throw (ex-info "CUDA registration requires its named CUDA artifact" {:reason :cuda-artifact-target})))
   (when (some #(contains? kernel %) [:arena-id ::cleanup/owner ::module])
     (throw (ex-info "Registration cannot import native lifetime fields" {:reason :invalid-kernel-registration})))
   (ensure-init!)
   (cleanup/with-registry-use kernel-registry
     (let [candidate (root/construct-child!
                      state []
                      (fn [_ entry] (assoc kernel :arena-id arena ::context (:context @(:projection entry))
                                          ::compiled (atom nil))) nil)]
       (cleanup/build!
        (::cleanup/owner candidate)
        #(do (cleanup/publish-replacement! kernel-registry [name] candidate
                                           (fn [old] (cleanup/release! (::cleanup/owner old))))
             candidate)
        #(cleanup/retain-owned-entry! kernel-registry candidate))))
   nil))

(defn kernel-registry-entry [name] (get @kernel-registry name))
(defn close-kernel-arena! [arena]
  (cleanup/with-registry-use kernel-registry
    (cleanup/release-entries! kernel-registry
                              (filterv #(= arena (:arena-id (second %))) @kernel-registry)
                              #(cleanup/release! (::cleanup/owner %)))) nil)

(defn- compiled! [registered]
  (cleanup/assert-live! (::cleanup/owner registered))
  (let [slot (::compiled registered)]
    (if @slot
      @slot
      (let [[major minor] (:compute-capability @state)
            arch (str "compute_" major minor)
            compiled (nvrtc/compile-artifact-ptx registered arch :include-paths (:include-paths @state))]
        (clojure.core/reset! slot compiled)
        compiled))))

(defn- parameter! [arena slot value]
  (let [scalar? (= :scalar (:kind slot))
        layout (if scalar? (case (:kernel-dtype slot)
                            :float ValueLayout/JAVA_FLOAT :double ValueLayout/JAVA_DOUBLE
                            :int ValueLayout/JAVA_INT :long ValueLayout/JAVA_LONG
                            :byte ValueLayout/JAVA_BYTE :half ValueLayout/JAVA_SHORT)
                   ValueLayout/JAVA_LONG)
        value (if scalar? (:value (call/validate-scalar-value! slot value)) (:device-pointer value))
        cast-value (if scalar? (case (:kernel-dtype slot)
                                :float (float value) :double (double value) :int (int value)
                                :long (long value) :byte (byte value) :half (short value)) (long value))
        segment (.allocate arena layout)]
    (.set segment layout (long 0) cast-value) segment))

(defn bind-kernel-call
  ([kernel-call] (bind-kernel-call kernel-call {}))
  ([kernel-call {:keys [adopt-cleanup! async?]}]
   (when async?
     (throw (ex-info "CUDA asynchronous prepared calls are unsupported"
                     {:reason :cuda-unsupported-operation :backend :cuda
                      :operation :bind-kernel-call :async? true})))
   (let [{:keys [kernel-name abi pairs pointer-pairs] :as plan} (call/binding-plan kernel-call)
         registered (or (get @kernel-registry kernel-name)
                        (throw (ex-info "Kernel not registered" {:kernel-name kernel-name})))
         _ (call/validate-registered! kernel-call registered)
         _ (doseq [[slot _] pairs]
             (when (and (= :scalar (:kind slot)) (= :half (:kernel-dtype slot)))
               (throw (ex-info "CUDA half scalar transport is unsupported"
                               {:reason :cuda-unsupported-operation :backend :cuda :operation :bind-scalar
                                :dtype :half}))))
         buffers (mapv (comp assert-buffer-live! second) pointer-pairs)
         _ (abi/validate-physical-pointer-dtypes! abi (mapv :dtype buffers))
         _ (call/validate-resident-output-capacities! kernel-call plan registered (fn [b _] (:n-elements b)))
         arena-slot (cleanup/acquisition-slot) module-slot (cleanup/acquisition-slot) token (random-uuid)
         pinned (atom []) graph-borrows (atom #{})]
     (cleanup/with-registry-use kernel-registry
       (cleanup/assert-registration-current! kernel-registry kernel-name registered)
       (root/construct-child!
        state [{:id :parameter-arena
                :release #(do
                            (assert-unborrowed! graph-borrows :cuda-prepared-in-use
                                                "CUDA prepared call is retained by a graph")
                            (cleanup/release-native! arena-slot (fn [a] (.close a))))}
               {:id :module :after #{:parameter-arena}
                :release #(cleanup/release-native! module-slot (fn [m] (native! :module-unload [m])))}
               {:id :bindings :after #{:module}
                :release #(doseq [borrow @pinned] (swap! borrow disj token))}]
        (fn [_ entry]
          (let [compiled (compiled! registered)
                module (cleanup/acquire-native! module-slot #(native! :module-load [(:ptx compiled)]))
                function (native! :module-function [module kernel-name])
                arena (cleanup/acquire-native! arena-slot #(Arena/ofShared))
                parameters (.allocate arena (long (* 8 (count pairs))) (long 8))]
            (doseq [borrow (distinct (map :borrows buffers))]
              (swap! pinned conj borrow) (swap! borrow conj token))
            (doseq [[index [slot value]] (map-indexed vector pairs)]
              (.set parameters ValueLayout/ADDRESS (long (* index 8)) (parameter! arena slot value)))
            {:function function :parameters parameters :geometry (:geometry kernel-call)
             :graph-borrows graph-borrows
             :kernel-call kernel-call :binding-plan plan :kernel-name kernel-name
             :compilation-identity (select-keys compiled [:nvrtc-version :virtual-architecture
                                                         :include-paths :virtual-header-version])
             ::context (:context @(:projection entry))}))
        adopt-cleanup!)))))

(defn- assert-prepared-live! [prepared]
  (cleanup/assert-live! (::cleanup/owner prepared))
  (doseq [[_ buffer] (get-in prepared [:binding-plan :pointer-pairs])] (assert-buffer-live! buffer))
  prepared)

(defn launch-registered-bound! [prepared]
  (assert-prepared-live! prepared)
  (native! :launch [(:function prepared) (:geometry prepared) (:parameters prepared) (:stream @state)])
  (native! :stream-sync [(:stream @state)]))

(defn destroy-prepared! [prepared]
  (assert-unborrowed! (:graph-borrows prepared) :cuda-prepared-in-use "CUDA prepared call is retained by a graph")
  (cleanup/release! (::cleanup/owner prepared)))
(defn record-graph!
  ([prepareds] (record-graph! prepareds {}))
  ([prepareds opts]
   (doseq [prepared prepareds] (assert-prepared-live! prepared))
   (execution/record-graph! state prepareds opts)))
(defn submit-graph! [graph]
  (doseq [prepared (:prepareds graph)] (assert-prepared-live! prepared))
  (execution/submit-graph! state graph native!))
(defn event-complete? [token] (execution/event-complete? token native!))
(defn await-event! [token] (execution/await-event! token native!))
(defn release-event! [token] (execution/release-event! token))
(defn destroy-graph! [graph] (execution/destroy-graph! graph))
(defn reset-graph-events! [graph] (execution/reset-graph-events! graph))
(defn read-graph-timestamps! [graph] (execution/read-graph-timestamps! graph native!))
(defn replay-graph! [graph]
  (let [token (submit-graph! graph)]
    (try (await-event! token) (finally (release-event! token)))))
(defn synchronize-async! [] (ensure-init!) (native! :stream-sync [(:stream @state)]))
(defn transfer-capabilities []
  {:submission :unsupported :host-staging :runtime-owned-native
   :independent-physical-queue? false :queue-ordering :in-order
   :async-h2d? false :async-d2h? false :peer-transfer? false :peer-mechanisms []
   :event-semantics :device-completion :host-lease-until-await? false
   :host-memory :any :physically-serialized? true})

(defn- unsupported! [operation]
  (throw (ex-info "CUDA runtime operation is unsupported"
                  {:reason :cuda-unsupported-operation :backend :cuda :operation operation})))
(defn invoke-registered-kernel [& _] (unsupported! :invoke-registered-kernel))
(defn invoke-registered-map-void-kernel [& _] (unsupported! :invoke-registered-map-void-kernel))
(defn bind-registered-map-void-kernel [& _] (unsupported! :bind-registered-map-void-kernel))
(defn register-kernel-dispatch! [& _] (unsupported! :register-kernel-dispatch!))
(defn kernel-dispatch-registry-entry [& _] (unsupported! :kernel-dispatch-registry-entry))
(defn submit-range-batch! [& _] (unsupported! :submit-range-batch!))
(defn copy-buffer-range! [& _] (unsupported! :copy-buffer-range!))
(defn shutdown! [] (root/shutdown-construction! state))
(defn reset! [] (shutdown!))
