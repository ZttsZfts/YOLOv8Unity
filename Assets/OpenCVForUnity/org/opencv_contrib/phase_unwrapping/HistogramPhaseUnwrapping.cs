#if !UNITY_WEBGL

using OpenCVForUnity.CoreModule;
using OpenCVForUnity.UtilsModule;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace OpenCVForUnity.Phase_unwrappingModule
{

    // C++: class HistogramPhaseUnwrapping
    /**
     * Class implementing two-dimensional phase unwrapping based on CITE: histogramUnwrapping
     * This algorithm belongs to the quality-guided phase unwrapping methods.
     * First, it computes a reliability map from second differences between a pixel and its eight neighbours.
     * Reliability values lie between 0 and 16*pi*pi. Then, this reliability map is used to compute
     * the reliabilities of "edges". An edge is an entity defined by two pixels that are connected
     * horizontally or vertically. Its reliability is found by adding the the reliabilities of the
     * two pixels connected through it. Edges are sorted in a histogram based on their reliability values.
     * This histogram is then used to unwrap pixels, starting from the highest quality pixel.
     *
     * The wrapped phase map and the unwrapped result are stored in CV_32FC1 Mat.
     */

    public class HistogramPhaseUnwrapping : PhaseUnwrapping
    {

        protected override void Dispose(bool disposing)
        {

            try
            {
                if (disposing)
                {
                }
                if (IsEnabledDispose)
                {
                    if (nativeObj != IntPtr.Zero)
                        phase_1unwrapping_HistogramPhaseUnwrapping_delete(nativeObj);
                    nativeObj = IntPtr.Zero;
                }
            }
            finally
            {
                base.Dispose(disposing);
            }

        }

        protected internal HistogramPhaseUnwrapping(IntPtr addr) : base(addr) { }

        // internal usage only
        public static new HistogramPhaseUnwrapping __fromPtr__(IntPtr addr) { return new HistogramPhaseUnwrapping(addr); }

        //
        // C++:  void cv::phase_unwrapping::HistogramPhaseUnwrapping::getInverseReliabilityMap(Mat& reliabilityMap)
        //

        /**
         * Get the reliability map computed from the wrapped phase map.
         *
         * param reliabilityMap Image where the reliability map is stored.
         */
        public void getInverseReliabilityMap(Mat reliabilityMap)
        {
            ThrowIfDisposed();
            if (reliabilityMap != null) reliabilityMap.ThrowIfDisposed();

            phase_1unwrapping_HistogramPhaseUnwrapping_getInverseReliabilityMap_10(nativeObj, reliabilityMap.nativeObj);


        }


#if (UNITY_IOS || UNITY_WEBGL) && !UNITY_EDITOR
        const string LIBNAME = "__Internal";
#else
        const string LIBNAME = "opencvforunity";
#endif



        // C++:  void cv::phase_unwrapping::HistogramPhaseUnwrapping::getInverseReliabilityMap(Mat& reliabilityMap)
        [DllImport(LIBNAME)]
        private static extern void phase_1unwrapping_HistogramPhaseUnwrapping_getInverseReliabilityMap_10(IntPtr nativeObj, IntPtr reliabilityMap_nativeObj);

        // native support for java finalize()
        [DllImport(LIBNAME)]
        private static extern void phase_1unwrapping_HistogramPhaseUnwrapping_delete(IntPtr nativeObj);

    }
}
#endif