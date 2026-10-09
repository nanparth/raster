(ns raster.gpu.cuda-numerical-device-test
  "Numerical qualification through public equation-first Compiled/LinkPlan execution."
  (:require [raster.gpu.device-probe :as device-probe]
            [clojure.test :refer [deftest is]]
            [raster.compiler.passes.parallel.full-reduction-device-test :as reduction]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.core :as gpu]
            [raster.gpu.cuda-driver-probe :as probe]
            [raster.gpu.cuda-runtime :as runtime]
            [raster.gpu.runtime-root :as root]
            [raster.gpu.value :as value]))

(defn- independent-dot [x y n]
  (reduce + 0.0 (map #(* (double %1) (double %2)) (take n x) (take n y))))

(defn- assert-result! [program plan actual x y n]
  (let [session (get-in program [:executable :session])
        state-node (get-in plan [:attributes :public-buffer-bindings 'out])
        physical-view (get-in program [:executable :node-views state-node])
        physical-result (float-array 2) expected (independent-dot x y n)]
    (is (value/device-array? actual))
    (is (= expected (double (first (value/->host actual)))))
    (is (= [2] (get-in physical-view [:view :shape])))
    (is (= 8 (get-in physical-view [:view :byte-length]))
        "certified caller state retains both physical elements")
    (gpu/download-range! session physical-view physical-result {:elements 2})
    (is (= [expected 456.0] (vec physical-result)) "out[1] remains the initialized guard")))

(defn- prove-dot! [n]
  (let [capacity (max 1 n)
        x (float-array (map #(float (- (mod % 7) 3)) (range capacity)))
        y (float-array (repeat capacity 0.5)) out (float-array [123 456])
        changed-x (float-array (map #(float (+ % 4)) x))
        changed-y (float-array (repeat capacity -0.25))
        prepared (compiled/lower #'reduction/dot-into! [x y out (long n)]
                                 {:compiler :equation-first :target :cuda:0 :dtype :float
                                  :outputs '[out] :roles {'out :state} :on-non-resident :throw})
        plan (compiled/plan prepared)
        program (compiled/instantiate! prepared) last-output (atom nil)]
    (try
      (is (compiled/prepared? prepared))
      (is (= :state (get-in plan [:attributes :public-buffer-roles 'out])))
      (let [first-output (:out (program {}))]
        (reset! last-output first-output)
        (assert-result! program plan first-output x y n)
        (let [replayed (:out (program {:x changed-x :y changed-y}))]
          (reset! last-output replayed)
          (is (not (value/live? first-output)) "changed-input replay invalidates prior outputs")
          (assert-result! program plan replayed changed-x changed-y n)))
      ;; Public resident inputs are copied D2D into the plan's owned input nodes. This proves
      ;; source-view offsets and padding; it does not claim the kernel pointers have these offsets.
      (let [source-session (gpu/make-session :cuda:0)
            padded-x (float-array (repeat (+ capacity 8) 37))
            padded-y (float-array (repeat (+ capacity 10) 61))]
        (System/arraycopy changed-x 0 padded-x 3 capacity)
        (System/arraycopy changed-y 0 padded-y 5 capacity)
        (try
          (gpu/alloc! source-session {:x [:float (alength padded-x) nil]
                                      :y [:float (alength padded-y) nil]})
          (gpu/upload! source-session :x padded-x)
          (gpu/upload! source-session :y padded-y)
          (let [x-view (gpu/buffer-view source-session :x {:byte-offset 12 :shape [capacity]})
                y-view (gpu/buffer-view source-session :y {:byte-offset 20 :shape [capacity]})
                x-value (value/wrap-external-view (gpu/buffer source-session :x) :cuda:0 (:view x-view))
                y-value (value/wrap-external-view (gpu/buffer source-session :y) :cuda:0 (:view y-view))
                prior @last-output
                replayed (with-redefs [value/->host (fn [& _]
                                                    (throw (ex-info "resident input was materialized on host" {})))]
                           (:out (program {:x x-value :y y-value})))]
            (reset! last-output replayed)
            (is (not (value/live? prior)))
            (assert-result! program plan replayed changed-x changed-y n)
            (is (= [12 20] (mapv #(get-in % [:view :byte-offset]) [x-view y-view])))
            (let [actual-x (gpu/download source-session :x)
                  actual-y (gpu/download source-session :y)]
              (is (= (vec padded-x) (vec actual-x)) "every source element and padding guard survives D2D")
              (is (= (vec padded-y) (vec actual-y)))))
          (finally (gpu/close-session! source-session))))
      (println "[CUDA PUBLIC REDUCTION]" {:n n :physical-state-shape [2] :guard 456.0
                                           :roles (get-in plan [:attributes :public-buffer-roles])
                                           :instances (count (:instances plan))
                                           :resident-source-view-offsets [12 20]
                                           :resident-input-transport :device-to-device
                                           :device (runtime/execution-device-info)})
      (finally (compiled/close! program)))
    (is (not (value/live? @last-output)))
    (is (= 0 (root/lease-count runtime/state)))))

(deftest public-generated-dot-state-guard-replay-and-resident-input-views
  (if (= "1" (System/getenv "RASTER_REQUIRE_CUDA_DEVICE"))
    (let [device (probe/probe)]
      (is (= :ready (:status device)) "required CUDA execution cannot skip")
      (when (= :ready (:status device))
        (doseq [n [8 0 1025]] (prove-dot! n))))
    (device-probe/cuda-skip! "Public numerical qualification requires RASTER_REQUIRE_CUDA_DEVICE=1" :not-requested)))

(defn- prove-generated-graph-offsets! [n]
  (let [capacity (max 1 n)
        x (float-array (map #(float (- (mod % 7) 3)) (range capacity)))
        y (float-array (repeat capacity 0.5))
        prepared (compiled/lower #'reduction/dot-into! [x y (float-array [123 456]) (long n)]
                                 {:compiler :equation-first :target :cuda:0 :dtype :float
                                  :outputs '[out] :roles {'out :state} :on-non-resident :throw})
        graph (get-in (compiled/plan prepared) [:instances 0 :call :steps 0 :graph])
        session (gpu/make-session :cuda:0)
        padded-x (float-array (repeat (+ capacity 8) 37))
        padded-y (float-array (repeat (+ capacity 10) 61))
        padded-out (float-array (repeat 11 313))]
    (System/arraycopy x 0 padded-x 3 capacity)
    (System/arraycopy y 0 padded-y 5 capacity)
    (aset-float padded-out 7 123)
    (aset-float padded-out 8 456)
    (try
      (gpu/alloc! session {:x [:float (alength padded-x) nil]
                           :y [:float (alength padded-y) nil]
                           :out [:float (alength padded-out) nil]})
      (gpu/upload! session :x padded-x)
      (gpu/upload! session :y padded-y)
      (gpu/upload! session :out padded-out)
      (let [x-view (gpu/buffer-view session :x {:byte-offset 12 :shape [capacity]})
            y-view (gpu/buffer-view session :y {:byte-offset 20 :shape [capacity]})
            out-view (gpu/buffer-view session :out {:byte-offset 28 :shape [2]})
            scalar-values (into {}
                                (for [{:keys [id dtype]} (:scalars graph)]
                                  (do
                                    (is (= 'n id) "the generated graph retains its public scalar ABI")
                                    [id {:type dtype :value (long n)}])))
            bound (gpu/bind-kernel-graph! session :dot graph
                                         {'x x-view 'y y-view 'out out-view} scalar-values)]
        (is (= [12 20 28] (mapv #(get-in % [:view :byte-offset]) [x-view y-view out-view])))
        (doseq [pass [0 1]]
          (let [next-x (float-array (map #(float (+ % (* pass 4))) x))
                next-y (float-array (repeat capacity (if (zero? pass) 0.5 -0.25)))
                expected (independent-dot next-x next-y n)]
            (gpu/upload-range! session x-view next-x {:elements capacity})
            (gpu/upload-range! session y-view next-y {:elements capacity})
            (with-redefs [value/->host (fn [& _]
                                       (throw (ex-info "generated graph materialized a host value" {})))]
              (gpu/run-kernel-graph! session bound))
            (System/arraycopy next-x 0 padded-x 3 capacity)
            (System/arraycopy next-y 0 padded-y 5 capacity)
            (aset-float padded-out 7 (float expected))
            (is (= (vec padded-x) (vec (gpu/download session :x))))
            (is (= (vec padded-y) (vec (gpu/download session :y))))
            (is (= (vec padded-out) (vec (gpu/download session :out)))
                "only out[0] in the offset two-element state changes; every guard remains intact")))
        (println "[CUDA GENERATED GRAPH OFFSET REDUCTION]"
                 {:n n :kernel-pointer-byte-offsets [12 20 28] :state-view-shape [2]
                  :guard 456.0 :graph-nodes (count (:nodes graph))
                  :route :equation-first-generated-graph-public-runtime-binding}))
      (finally (gpu/close-session! session)))
    (is (= 0 (root/lease-count runtime/state)))))

(deftest generated-equation-graph-binds-actual-input-and-state-offsets
  (if (= "1" (System/getenv "RASTER_REQUIRE_CUDA_DEVICE"))
    (let [device (probe/probe)]
      (is (= :ready (:status device)) "required CUDA execution cannot skip")
      (when (= :ready (:status device))
        (doseq [n [8 0 1025]] (prove-generated-graph-offsets! n))))
    (device-probe/cuda-skip! "Generated graph offset qualification requires RASTER_REQUIRE_CUDA_DEVICE=1" :not-requested)))
