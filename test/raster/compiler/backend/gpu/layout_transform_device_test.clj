(ns raster.compiler.backend.gpu.layout-transform-device-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.gpu.kernel-body-opencl :as emitter]
            [raster.compiler.backend.gpu.kernel-body-target :as target]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.scheduled-kernel-body :as scheduled]
            [raster.compiler.ir.kernel-call :as call]
            [raster.compiler.passes.parallel.layout-transform-schedule :as schedule]
            [raster.gpu.device-probe :as probe]))

(defn- layout-artifact [kernel]
  (let [parameters (:parameters kernel)
        arguments (mapv :id parameters)
        names (zipmap arguments ["x" "y" "rows" "cols"])
        schedule (scheduled/make
                   {:source {:operation :layout-transform :id (:id kernel)}
                    :body kernel :arguments arguments
                    :preconditions (body/required-preconditions kernel)
                    :scalar-bindings (scheduled/derive-scalar-bindings kernel arguments)
                    :effects {:kind :layout-transform
                              :uses (scheduled/derive-uses kernel arguments)}
                    :legality {:kind :dense-affine-layout-transform}
                    :numerics {:mode :exact :policy :layout-permutation}})]
    (target/emit-artifact "layout_width_probe" schedule :opencl-portable
                          {:parameter-names names})))

(defn- runtime-function [name]
  (ns-resolve (find-ns 'raster.gpu.ocl-runtime) name))

(defn- execute
  [kernel input initial-output scalars]
  (let [compiled (layout-artifact kernel)
        scalar-arguments (mapv (fn [parameter value]
                                {:type (:dtype parameter) :value value})
                              (drop 2 (:parameters kernel)) scalars)
        ;; Check actual ABI scalars before buffer allocation; placeholders are never bound.
        _ (call/validate-preconditions! compiled (into [::input ::output] scalar-arguments))
        x ((runtime-function 'buffer-of-array) input :float)]
    (try
      (let [y ((runtime-function 'buffer-of-array) initial-output :float)]
        (try
          ((runtime-function 'register-kernel!) (:kernel-name compiled) compiled)
          (let [prepared ((runtime-function 'bind-kernel-call)
                          (call/make compiled (into [x y] scalar-arguments)))]
            (try
              ((runtime-function 'launch-registered-bound!) prepared)
              ((runtime-function 'buffer->array) y)
              (finally ((runtime-function 'destroy-prepared!) prepared))))
          (finally ((runtime-function 'free-buffer!) y))))
      (finally ((runtime-function 'free-buffer!) x)))))

(deftest layout-device-helper-retains-conditions-before-allocation
  (let [kernel (schedule/transpose-body
                {:id :transpose-preflight :input 'x :output 'y :element-dtype :float
                 :row-extent-dtype :long :column-extent-dtype :long})
        compiled (layout-artifact kernel)
        conditions (body/required-preconditions kernel)
        names (zipmap (map :id (:parameters kernel)) ["x" "y" "rows" "cols"])
        module (emitter/emit-scalar-module "layout_width_probe" kernel
                                          {:target-dialect :opencl-portable :parameter-names names})
        contacts (atom [])]
    (is (= conditions (:preconditions compiled) (:required-preconditions module)))
    (is (= (or (:compilation module) {}) (artifact/compilation compiled)))
    (is (= (mapv (juxt :id :kind :dtype) (:parameters kernel))
           (mapv (juxt :name :kind :kernel-dtype) (:abi compiled))))
    (is (= (mapv :id (:parameters kernel)) (:arguments compiled)))
    (doseq [condition conditions]
      (is (= :kernel-artifact-required-preconditions
             (try (artifact/validate! (assoc compiled :preconditions
                                             (vec (remove #{condition} conditions)))) :accepted
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
    (is (= :discarded-precondition-contract
           (try (emitter/emit-scalar-kernel "layout_width_probe" kernel
                  {:target-dialect :opencl-portable :parameter-names names}) :accepted
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (doseq [shape [[2 3] [0 3] [2 0]]]
      (is (call/validate-preconditions! compiled
            (into [::input ::output] (map #(hash-map :type :long :value %) shape)))))
    (with-redefs [runtime-function (fn [name]
                                   (swap! contacts conj name)
                                   (throw (ex-info "invalid shape reached runtime" {:operation name})))]
      (doseq [shape [[-1 3] [2 -1] [-1 -1]]]
        (is (= :kernel-precondition-failed
               (try (execute kernel (float-array 0) (float-array 0) shape) :accepted
                    (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
      (is (thrown? ArithmeticException
            (execute kernel (float-array 0) (float-array 0) [Long/MAX_VALUE 2]))))
    (is (empty? @contacts) "invalid scalar domains fail before resolving any allocation/runtime operation")))

(deftest layout-integer-widths-execute-with-masked-tails
  (if-not @probe/opencl-available?
    (probe/opencl-skip! "layout transform retained integer widths")
    (do
      (doseq [extent-dtype [:int :long]]
        (let [n 1025
              input (float-array (map float (range n)))
              kernel (schedule/cast-body
                      {:id :cast-width :input 'x :output 'y
                       :extent-dtype extent-dtype
                       :source-dtype :float :destination-dtype :float
                       :vector-width 4 :rounding :exact :overflow :exact})
              actual (vec (execute kernel input (float-array (repeat (+ n 3) -1.0)) [n]))]
          (is (= (vec input) (subvec actual 0 n)))
          (is (= [-1.0 -1.0 -1.0] (subvec actual n)))))
      (doseq [row-dtype [:int :long] column-dtype [:int :long]]
        (let [kernel (schedule/transpose-body
                      {:id :transpose-width :input 'x :output 'y :element-dtype :float
                       :row-extent-dtype row-dtype :column-extent-dtype column-dtype})
              actual (vec (execute kernel (float-array [1 2 3 4 5 6])
                                   (float-array (repeat 8 -1.0)) [2 3]))]
          (is (= [1.0 4.0 2.0 5.0 3.0 6.0] (subvec actual 0 6)))
          (is (= [-1.0 -1.0] (subvec actual 6))))))))
