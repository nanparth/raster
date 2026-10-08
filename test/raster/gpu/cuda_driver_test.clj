(ns raster.gpu.cuda-driver-test
  (:require [clojure.test :refer [deftest is]]
            [raster.gpu.cuda-driver :as driver])
  (:import [java.lang.foreign MemorySegment ValueLayout]))

(defn- with-raw [f]
  (let [calls (atom []) fault (atom {}) context (MemorySegment/ofAddress 11)
        raw (fn [symbol _ args]
              (swap! calls conj symbol)
              (when-let [error (get @fault symbol)] (throw error))
              (case symbol
                "cuDevicePrimaryCtxRetain" (.set (first args) ValueLayout/ADDRESS (long 0) context)
                "cuStreamCreate" (.set (first args) ValueLayout/ADDRESS (long 0) (MemorySegment/ofAddress 12))
                "cuDeviceGetAttribute" (.set (first args) ValueLayout/JAVA_INT (long 0)
                                            (int (get {1 1024 8 49152 10 32 16 40} (second args) 1)))
                "cuCtxPopCurrent_v2" (.set (first args) ValueLayout/ADDRESS (long 0) context)
                "cuMemAlloc_v2" (.set (first args) ValueLayout/JAVA_LONG (long 0) (long 1024))
                nil))]
    (binding [driver/*invoke* nil driver/*context* context]
      (with-redefs-fn {(ns-resolve 'raster.gpu.cuda-driver 'raw!) raw}
        #(f calls fault context)))))

(deftest each-thread-operation-restores-the-exact-previous-context
  (with-raw
    (fn [calls _ _]
      (is (= 1024 (driver/invoke! :allocate [4])))
      (is (= ["cuCtxPushCurrent_v2" "cuMemAlloc_v2" "cuCtxPopCurrent_v2"] @calls)))))

(deftest primary-native-failure-preserves-context-pop-failure-as-suppressed
  (with-raw
    (fn [calls fault _]
      (let [primary (ex-info "allocation failed" {}) secondary (ex-info "context pop failed" {})]
        (reset! fault {"cuMemAlloc_v2" primary "cuCtxPopCurrent_v2" secondary})
        (is (identical? primary (try (driver/invoke! :allocate [4]) nil (catch Throwable e e))))
        (is (= [secondary] (vec (.getSuppressed primary))))
        (is (= "cuCtxPopCurrent_v2" (last @calls)))))))

(deftest primary-context-release-occurs-after-leaving-its-thread-context
  (with-raw
    (fn [calls _ context]
      (driver/invoke! :destroy-runtime [{:context context :stream (MemorySegment/ofAddress 12) :device 0}])
      (is (= ["cuCtxPushCurrent_v2" "cuStreamDestroy_v2" "cuCtxPopCurrent_v2" "cuDevicePrimaryCtxRelease_v2"] @calls)))))

(deftest initialization-records-queried-device-limits-and-api-version
  (with-raw
    (fn [_ _ _]
      (let [info (driver/invoke! :initialize [0])]
        (is (= {:max-workgroup-size 1024 :shared-local-memory 49152 :warp-size 32
                :multiprocessor-count 40}
               (select-keys info [:max-workgroup-size :shared-local-memory :warp-size :multiprocessor-count])))
        (is (contains? info :driver-api-version))))))
