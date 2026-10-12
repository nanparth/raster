(ns raster.compiler.passes.parallel.product-consumer-body-test
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.kernel-body-opencl :as emitter]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.kernel-precondition :as precondition]
            [raster.compiler.passes.parallel.product-consumer-body :as candidate]
            [raster.compiler.passes.parallel.product-consumer-region-test :as fixtures]))

(defn- candidate-body []
  (let [program (#'fixtures/scheduled-product-consumer)]
    (:kernel-body
     (candidate/from-program program (#'fixtures/numerical-equations program)))))

(defn- pair-candidate-body []
  (let [program (#'fixtures/scheduled-product-pair-consumer)]
    (:kernel-body
     (candidate/from-program program (#'fixtures/numerical-equations program)))))

(defn- operation-kinds [kernel-body]
  (->> (:operations kernel-body)
       (tree-seq coll? seq)
       (keep #(some-> % class .getSimpleName))
       set))

(deftest product-and-ordered-consumer-share-one-validated-body
  (let [kernel-body (candidate-body)]
    (is (identical? kernel-body (body/validate! kernel-body)))
    (is (= :product-tree-ordered-consumer (get-in kernel-body [:schedule :strategy])))
    (is (= [8] (get-in kernel-body [:launch :workgroup-size])))
    (is (= ['input 'weights 'output 'rows] (mapv :id (:parameters kernel-body))))
    (is (= ['partials] (mapv :id (:allocations kernel-body)))
        "the graph-private intermediate becomes workgroup storage, not a public allocation")
    (is (every? (operation-kinds kernel-body)
                ["ForLoop" "WorkgroupBarrier" "ScalarLoad" "ScalarStore"]))
    (is (= :implementation-defined
           (get-in kernel-body [:attributes :inner-numerics :association])))
    (is (= :ordered (get-in kernel-body [:attributes :outer-numerics :association])))))

(deftest one-body-projects-to-portable-opencl-cuda-and-hip
  (let [kernel-body (candidate-body)]
    (doseq [[dialect marker]
            [[:opencl-portable ["__kernel void product_consumer" "barrier"]]
             [:cuda ["extern \"C\" __global__ void product_consumer" "__syncthreads"]]
             [:hip ["extern \"C\" __global__ void product_consumer" "__syncthreads"]]]]
      (testing (name dialect)
        (let [module (emitter/emit-scalar-module
                      "product_consumer" kernel-body {:target-dialect dialect})]
          (is (every? #(str/includes? (:source module) %) marker))
          (is (not (str/includes? (:source module) "partials,"))
              "private scratch must not leak into the public ABI"))))))

(deftest product-valued-workgroup-fallback-keeps-independent-typed-scratch
  (let [kernel-body (pair-candidate-body)]
    (is (identical? kernel-body (body/validate! kernel-body)))
    (is (= '[left-partials right-partials] (mapv :id (:allocations kernel-body))))
    (is (= [:int :int] (mapv :dtype (:allocations kernel-body))))
    (is (= 64 (get-in kernel-body [:launch :shared-memory-bytes])))))


(deftest nested-positive-consumer-scopes-preserve-divisor-and-prefix-obligations
  ;; Arithmetic-only pointer fixture; x[1] is not a storage-capacity certificate.
  (let [parameters [(body/->KernelParameter 'rows :scalar :long [] nil nil :parameter)
                    (body/->KernelParameter 'columns :scalar :long [] nil nil :parameter)
                    (body/->KernelParameter 'scale :scalar :long [] nil nil :parameter)]
        indices [(body/->IndexBinding 'group :group 0)]
        actual-launch (launch/spec {:workgroup-size [8] :group-count [1]})
        positive (fn [id bound]
                   (body/->ScalarCompute (body/value id :predicate)
                      (body/scalar-expression :lt :predicate [(body/literal 0 :long) bound])))
        outer-condition (positive 'rows-positive 'rows)
        inner-condition (positive 'columns-positive 'columns)
        definitions [(body/->IndexCompute 'row
                        (body/expression :mod (body/index-cast 'group :long :exact) 'rows))]
        header (body/->ForLoop (body/value 'column :long) (body/index-cast 0 :long :exact)
                              'columns 1 [] [(body/->Yield [])] [] {})
        inner (body/->IfRegion 'columns-positive (conj definitions header (body/->Yield []))
                              [(body/->Yield [])] [])
        outer (body/->IfRegion 'rows-positive [inner (body/->Yield [])] [(body/->Yield [])] [])
        region {:parameters parameters :indices indices :launch actual-launch
                :condition-definition inner-condition :if-region inner :index-definitions definitions
                :enclosing-if-regions [{:condition-definition outer-condition :if-region outer}]
                :inner-loop header}
        replay (body/scheduled-index-region-info region)
        values (:values replay)
        scaled (body/expression :mul 'row (body/index-cast 64 :long :exact))
        scaled-info (body/conditional-index-info scaled values)
        requirements (vec (distinct (concat (:requirements replay) (:required-preconditions scaled-info))))]
    (is (= 1 (get-in values ['rows :range :lower])))
    (is (= 1 (get-in values ['columns :range :lower])))
    (is (= 2 (launch/resolve-expression {'rows 3} (get-in values ['row :conditional-integer :upper]))))
    (is (= 0 (launch/resolve-expression {'rows Long/MIN_VALUE}
                                       (get-in values ['row :conditional-integer :upper]))))
    (is (not-any? #(= 'row (:index %)) (:guarded-index-domains (meta values)))
        "positive divisor bounds do not invent hardware-row activation identities")
    (is (empty? requirements)
        "the real hardware numerator bound proves row*64 without a loose rows capacity cap")
    (is (true? (precondition/check! requirements {'rows 3 'columns 8})))
    (is (true? (precondition/check! requirements {'rows Long/MIN_VALUE 'columns 0})))
    (is (true? (precondition/check! requirements {'rows Long/MAX_VALUE 'columns 8})))
    (let [unsafe-info (body/conditional-index-info (body/expression :mul 'row 'scale) values)
          caps (:required-preconditions unsafe-info)]
      (is (seq caps))
      (is (true? (precondition/check! caps {'rows Long/MAX_VALUE 'columns 8 'scale 64})))
      (is (thrown? ArithmeticException
                   (precondition/check! caps {'rows 3 'columns 8 'scale Long/MAX_VALUE}))))
    (doseq [changed [(assoc-in region [:enclosing-if-regions 0 :if-region :then-operations]
                              [(body/->Yield [])])
                     (assoc region :guard (body/->Guard :absent []))]]
      (let [outcome (try (body/scheduled-index-region-info changed) {:outcome :accepted}
                         (catch clojure.lang.ExceptionInfo error {:outcome :rejected :data (ex-data error)}))]
        (is (= :rejected (:outcome outcome)) (pr-str outcome))
        (is (= :kernel-body-index-region-condition (get-in outcome [:data :reason])) (pr-str outcome))))
    (let [ordinary (:values (body/initial-scheduled-index-info
                             {:parameters parameters :indices indices :launch actual-launch}))]
      (is (nil? (body/conditional-index-info (:expression (first definitions)) ordinary))
          "the same division outside the actual positive branch retains its invalid-divisor uncertainty"))))
