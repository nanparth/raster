(ns raster.gpu.opencl-retained-rounding-device-test
  "Normal public equation-first OpenCL compilation retains separate FP32 rounding."
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.arrays :as arrays]
            [raster.numeric :as numeric]
            [raster.math :as math]
            [raster.par :as par]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.device-probe :as probe]
            [raster.gpu.value :as value]))

(deftm rounding-control!
  [a :- (Array float) b :- (Array float) c :- (Array float)
   separate :- (Array float) fused :- (Array float) n :- Long] :- Void
  (par/map-void! i n
    (do
      (arrays/aset separate i (numeric/+ (numeric/* (arrays/aget a i) (arrays/aget b i))
                                         (arrays/aget c i)))
      (arrays/aset fused i (math/fma (arrays/aget a i) (arrays/aget b i) (arrays/aget c i))))))

(deftest public-opencl-retains-rounding-with-explicit-fma
  (if (= "1" (System/getenv "RASTER_REQUIRE_OPENCL_DEVICE"))
    (do
      (is @probe/opencl-available? "Required OpenCL device must be available")
      (when @probe/opencl-available?
        (let [a (float-array [(+ 1.0 (Math/scalb 1.0 (int -13)))])
              b (float-array [(- 1.0 (Math/scalb 1.0 (int -13)))])
              program (compiled/instantiate!
                       (compiled/lower #'rounding-control!
                                       [a b (float-array [-1]) (float-array [123])
                                        (float-array [456]) 1]
                                       {:compiler :equation-first :target :ocl:0 :dtype :float
                                        :outputs '[separate fused]
                                        :roles {'separate :state 'fused :state}
                                        :schedule {:precision :f32-scalar}
                                        :on-non-resident :throw}))]
          (try
            (let [outputs (program {})
                  bits (fn [key]
                         (Float/floatToRawIntBits
                          (float (first (value/->host (get outputs key))))))]
              ;; These dyadic terms are exact in binary64. Independent separate rounding is
              ;; +0; a single rounded exact product-plus-add is -2^-26.
              (is (= 0 (bits :separate)))
              (is (= (Float/floatToRawIntBits (float (- (Math/scalb 1.0 (int -26)))))
                     (bits :fused))))
            (finally (compiled/close! program))))))
    (is true "OpenCL native acceptance is opt-in with RASTER_REQUIRE_OPENCL_DEVICE=1")))
