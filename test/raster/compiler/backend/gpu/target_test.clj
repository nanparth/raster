(ns raster.compiler.backend.gpu.target-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.target :as target]))

(deftest descriptors-select-source-dialects-without-claiming-runtime-support
  (is (= :cuda (target/kernel-body-c-dialect
                {:device-type :gpu :backend :cuda :vendor "NVIDIA"})))
  (is (= :hip (target/kernel-body-c-dialect
               {:device-type :gpu :backend :hip :vendor "AMD"})))
  (is (= :opencl-intel (target/kernel-body-c-dialect
                        {:device-type :gpu :backend :ze :vendor "Intel"})))
  (is (= :opencl-portable (target/kernel-body-c-dialect
                           {:device-type :gpu :backend :opencl :vendor "AMD"
                            :subgroup-dialect :opencl-portable})))
  (testing "vendor names do not override an explicitly different runtime backend"
    (is (= :cuda (target/kernel-body-c-dialect
                  {:device-type :gpu :backend :cuda :vendor "Intel"
                   :matrix {:family :dpas}}))))
  (testing "generic non-Intel OpenCL must prove portable subgroup support explicitly"
    (is (nil? (target/kernel-body-c-dialect
               {:device-type :gpu :backend :opencl :vendor "AMD"}))))
  (is (nil? (target/kernel-body-c-dialect {:device-type :cpu :backend :cuda}))))

(deftest canonical-source-family-does-not-grant-subgroup-capabilities
  (let [descriptor {:device-type :gpu :backend :ocl :vendor "NVIDIA"}]
    (is (nil? (target/kernel-body-c-dialect descriptor)))
    (is (= :opencl-portable (target/source-dialect :ocl:0 :ocl descriptor)))
    (is (not (target/intel-opencl-subgroup-dialect? descriptor))))
  (doseq [[device backend expected] [[:ze:0 :ze :opencl-intel]
                                    [:cuda:0 :cuda :cuda]
                                    [:hip:0 :hip :hip]
                                    [:unknown:0 :unknown nil]]]
    (is (= expected (target/source-dialect device backend nil)))))
