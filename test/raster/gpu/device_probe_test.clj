(ns raster.gpu.device-probe-test
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [raster.gpu.device-probe :as probe]))

(defn- api
  [& {:keys [load! query-devices init! selected-device-info]
      :or {load! (constantly true)
           query-devices (constantly [{:name "enumerated"}])
           init! (constantly true)
           selected-device-info (constantly {:name "selected"})}}]
  {:load! load!
   :query-devices query-devices
   :init! init!
   :selected-device-info selected-device-info})

(deftest opencl-probe-distinguishes-absence-from-breakage
  (is (= :load-failed
         (:status
          (probe/probe-opencl-with
           (api :load! #(throw (Exception. "load")))))))
  (is (= :probe-error
         (:status
          (probe/probe-opencl-with
           (api :query-devices #(throw (Exception. "query")))))))
  (let [initialized? (atom false)
        result (probe/probe-opencl-with
                (api :query-devices (constantly [])
                     :init! #(reset! initialized? true)))]
    (is (= :no-device (:status result)))
    (is (false? @initialized?) "absence must not initialize a context")))

(deftest capability-gates-use-the-exact-selected-device
  (let [selected {:name "selected-without-fp16" :extensions "cl_khr_fp64"}
        result (probe/probe-opencl-with
                (api :query-devices
                     (constantly [{:name "other" :extensions "cl_khr_fp16"} selected])
                     :selected-device-info (constantly selected)))]
    (is (= :available (:status result)))
    (is (= selected (:device result)))
    (is (true? (probe/capability-supported? (:device result) :fp64)))
    (is (false? (probe/capability-supported? (:device result) :fp16))))
  (is (probe/capability-supported?
       {:extensions "cl_khr_fp64 cl_khr_fp16 cl_khr_subgroups"} :fp16)))

(deftest fp64-gate-does-not-confuse-precision-capabilities
  (is (probe/capability-supported? {:extensions "cl_amd_fp64"} :fp64))
  (is (not (probe/capability-supported? {:extensions "cl_khr_fp16"} :fp64))))

(defn- capture-skip
  [status expected?]
  (let [events (atom [])
        skip-log (var-get (ns-resolve 'raster.gpu.device-probe 'opencl-skip-log))
        prior-log @skip-log]
    (try
      (binding [probe/*expect-opencl?* expected?]
        (with-redefs [probe/opencl-status (delay status)
                      clojure.test/report (fn [event]
                                            (swap! events conj (:type event)))]
          (probe/opencl-skip! "probe-test")))
      @events
      (finally
        (reset! skip-log prior-log)))))

(deftest expected-opencl-mode-turns-absence-and-probe-errors-red
  (testing "ordinary CPU-only CI records a visible passing marker"
    (is (= [:pass] (capture-skip {:status :no-device} false))))
  (testing "an expected OpenCL lane fails on absence"
    (is (= [:fail] (capture-skip {:status :no-device} true))))
  (testing "an expected OpenCL lane fails on driver/query errors"
    (is (= [:fail]
           (capture-skip {:status :probe-error :error (Exception. "driver")} true)))))

(defn- capture-native-report
  [status required?]
  (let [events (atom [])
        reporter (ns-resolve 'raster.gpu.device-probe 'native-skip!)
        log-var (ns-resolve 'raster.gpu.device-probe 'native-skip-log)
        log (when log-var (var-get log-var))
        prior (when log @log)]
    (try
      (when reporter
        (with-redefs [clojure.test/report #(swap! events conj %)]
          (reporter status "structured-native-control" required?)))
      @events
      (finally (when log (reset! log prior))))))

(deftest structured-native-gates-preserve-required-and-optional-lanes
  (is (ifn? (ns-resolve 'raster.gpu.device-probe 'native-skip!))
      "native gates must use the shared structured reporting boundary")
  (doseq [status [:not-requested :no-device :unavailable :probe-error :missing-toolchain]]
    (let [optional (capture-native-report {:backend :cuda :status status} false)
          required (capture-native-report {:backend :cuda :status status} true)]
      (is (= [:pass] (mapv :type optional)) (str "optional " status " is one visible marker"))
      (is (= [:fail] (mapv :type required)) (str "required " status " cannot skip"))
      (is (some #(str/includes? (str (:message %)) "structured-native-control") optional)
          "structured markers identify the test rather than silently count success")))
  (doseq [required? [false true]]
    (is (= [:fail]
           (mapv :type (capture-native-report
                        {:backend :cuda :status :load-failed :error (Exception. "broken load")}
                        required?)))
        "a broken namespace load fails even on an optional lane")))

(deftest native-report-cannot-hide-an-available-or-unknown-state
  (when (ns-resolve 'raster.gpu.device-probe 'native-skip!)
    (doseq [status [:available :unknown]]
      (is (thrown? clojure.lang.ExceptionInfo
                   (capture-native-report {:backend :cuda :status status} false))
          "only an actual classified gate result may report a skip"))))

(deftest cuda-gate-classification-retains-the-original-driver-result
  (let [classify (ns-resolve 'raster.gpu.device-probe 'probe-cuda-with)]
    (is (ifn? classify) "optional probing must distinguish a driver query from namespace loading")
    (when classify
      (let [device {:ordinal 0 :name "selected" :compute-capability [8 6]}
            result (classify {:load! (constantly true)
                              :probe! (constantly {:status :ready :devices [device]
                                                   :driver-api-version 13020})})]
        (is (= :available (:status result)))
        (is (= [device] (:devices result)))
        (is (= 13020 (:driver-api-version result))))
      (doseq [status [:no-device :unavailable :driver-error]]
        (is (= (if (= :driver-error status) :probe-error status)
               (:status (classify {:load! (constantly true)
                                   :probe! (constantly {:status status :devices []})})))))
      (let [queried (atom false)
            result (classify {:load! #(throw (Exception. "load"))
                              :probe! #(reset! queried true)})]
        (is (= :load-failed (:status result)))
        (is (false? @queried) "a namespace failure cannot turn into absent hardware")))))

(deftest intel-subgroup-gate-inspects-the-selected-extension
  (is (true? (probe/capability-supported? {:extensions "cl_intel_subgroups"} :intel-subgroups)))
  (is (false? (probe/capability-supported? {:extensions "cl_khr_subgroups"} :intel-subgroups)))
  (is (false? (probe/capability-supported? {:extensions ""} :intel-subgroups))))
