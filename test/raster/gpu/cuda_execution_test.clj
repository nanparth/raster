(ns raster.gpu.cuda-execution-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.kernel-abi :as abi]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-call :as call]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.gpu.cuda-driver :as driver]
            [raster.gpu.cuda-nvrtc :as nvrtc]
            [raster.gpu.cuda-runtime :as runtime]
            [raster.gpu.resource-cleanup :as cleanup])
  (:import [java.lang.foreign MemorySegment]))

(defn- kernel []
  (artifact/make {:kernel-name "completion_contract" :target :cuda-c
                  :source "extern \"C\" __global__ void completion_contract(float* x) {}"
                  :abi [(abi/slot 'x :inout :float)] :arguments '[x]
                  :launch (launch/spec {:workgroup-size [1] :group-count [1]})}))

(defn- with-driver [f]
  (let [operations (atom []) ready (atom false) fault (atom nil) address (atom 4096)
        fake (fn [op args]
               (swap! operations conj [op args])
               (when (= op (:operation @fault)) (throw (:error @fault)))
               (case op
                 :initialize {:context (MemorySegment/ofAddress 1) :stream (MemorySegment/ofAddress 2)
                              :device 0 :compute-capability [8 6]}
                 :allocate (swap! address + 4096)
                 (:module-load :module-function :event-create) (MemorySegment/ofAddress (swap! address + 4096))
                 :event-query @ready
                 :event-sync (reset! ready true)
                 :event-elapsed 100.0
                 nil))]
    (binding [driver/*invoke* fake]
      (with-redefs [runtime/state (atom {}) runtime/kernel-registry (atom {})
                    nvrtc/compile-artifact-ptx (fn [& _] {:ptx (byte-array [1 0])})]
        (f operations ready fault)))))

(deftest graph-submit-poll-await-and-release-have-distinct-contracts
  (let [record! (ns-resolve 'raster.gpu.cuda-runtime 'record-graph!)]
    (is (some? record!) "CUDA logical graph replay must complete through native events")
    (when record!
      (with-driver
        (fn [operations ready _]
          (let [a (kernel) b (runtime/make-buffer 4 :float)]
            (runtime/register-kernel! (:kernel-name a) a)
            (let [prepared (runtime/bind-kernel-call (call/make a [b]))
                  graph (record! [prepared prepared])]
              (try
                (let [submit! (ns-resolve 'raster.gpu.cuda-runtime 'submit-graph!)
                      token (submit! graph)]
                  (is (= 2 (count (filter #(= :launch (first %)) @operations))))
                  (is (not @ready))
                  (is (false? ((ns-resolve 'raster.gpu.cuda-runtime 'event-complete?) token)))
                  (is (not-any? #(= :stream-sync (first %)) @operations) "submit/poll must not wait")
                  (is (thrown? clojure.lang.ExceptionInfo (submit! graph)) "one outstanding submission")
                  ((ns-resolve 'raster.gpu.cuda-runtime 'await-event!) token)
                  (is @ready)
                  ((ns-resolve 'raster.gpu.cuda-runtime 'release-event!) token)
                  (is (empty? (cleanup/pending (::cleanup/owner token))))
                  (reset! ready false)
                  ((ns-resolve 'raster.gpu.cuda-runtime 'release-event!) (submit! graph))
                  (is (some #(= :stream-sync (first %)) @operations) "release establishes completion"))
                (finally
                  ((ns-resolve 'raster.gpu.cuda-runtime 'destroy-graph!) graph)
                  (runtime/destroy-prepared! prepared)
                  (runtime/free-buffer! b)
                  (runtime/close-kernel-arena! nil))))))))))

(deftest empty-profiled-graphs-return-complete-without-creating-a-native-event
  (with-driver
    (fn [ops _ _]
      (runtime/initialize!)
      (let [before (count @ops) graph (runtime/record-graph! [] {:profile? true})]
        (try
          (let [token (runtime/submit-graph! graph)]
            (is (true? (:complete? token)))
            (is (true? (runtime/event-complete? token)))
            (is (nil? (runtime/await-event! token)))
            (is (nil? (runtime/release-event! token)))
            (is (= before (count @ops)) "empty graph completion has no native work"))
          (is (thrown-with-msg? clojure.lang.ExceptionInfo #"no profiled submission"
                                (runtime/read-graph-timestamps! graph)))
          (finally (runtime/destroy-graph! graph)))))))
