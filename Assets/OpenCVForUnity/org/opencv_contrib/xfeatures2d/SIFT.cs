
//using OpenCVForUnity.CoreModule;
//using OpenCVForUnity.Features2dModule;
//using OpenCVForUnity.UtilsModule;
//using System;
//using System.Collections.Generic;
//using System.Runtime.InteropServices;

//namespace OpenCVForUnity.Xfeatures2dModule
//{

//    // C++: class SIFT
//    /**
//     * Class for extracting keypoints and computing descriptors using the Scale Invariant Feature Transform
//     * (SIFT) algorithm by D. Lowe CITE: Lowe04 .
//     */

//    public class SIFT : Feature2D
//    {

//        protected override void Dispose(bool disposing)
//        {

//            try
//            {
//                if (disposing)
//                {
//                }
//                if (IsEnabledDispose)
//                {
//                    if (nativeObj != IntPtr.Zero)
//                        xfeatures2d_SIFT_delete(nativeObj);
//                    nativeObj = IntPtr.Zero;
//                }
//            }
//            finally
//            {
//                base.Dispose(disposing);
//            }

//        }

//        protected internal SIFT(IntPtr addr) : base(addr) { }

//        // internal usage only
//        public static new SIFT __fromPtr__(IntPtr addr) { return new SIFT(addr); }

//        //
//        // C++: static Ptr_SIFT cv::xfeatures2d::SIFT::create(int nfeatures = 0, int nOctaveLayers = 3, double contrastThreshold = 0.04, double edgeThreshold = 10, double sigma = 1.6)
//        //

//        /**
//         * param nfeatures The number of best features to retain. The features are ranked by their scores
//         *     (measured in SIFT algorithm as the local contrast)
//         *
//         *     param nOctaveLayers The number of layers in each octave. 3 is the value used in D. Lowe paper. The
//         *     number of octaves is computed automatically from the image resolution.
//         *
//         *     param contrastThreshold The contrast threshold used to filter out weak features in semi-uniform
//         *     (low-contrast) regions. The larger the threshold, the less features are produced by the detector.
//         *
//         *     param edgeThreshold The threshold used to filter out edge-like features. Note that the its meaning
//         *     is different from the contrastThreshold, i.e. the larger the edgeThreshold, the less features are
//         *     filtered out (more features are retained).
//         *
//         *     param sigma The sigma of the Gaussian applied to the input image at the octave \#0. If your image
//         *     is captured with a weak camera with soft lenses, you might want to reduce the number.
//         * return automatically generated
//         */
//        public static SIFT create(int nfeatures, int nOctaveLayers, double contrastThreshold, double edgeThreshold, double sigma)
//        {


//            return SIFT.__fromPtr__(xfeatures2d_SIFT_create_10(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma));


//        }

//        /**
//         * param nfeatures The number of best features to retain. The features are ranked by their scores
//         *     (measured in SIFT algorithm as the local contrast)
//         *
//         *     param nOctaveLayers The number of layers in each octave. 3 is the value used in D. Lowe paper. The
//         *     number of octaves is computed automatically from the image resolution.
//         *
//         *     param contrastThreshold The contrast threshold used to filter out weak features in semi-uniform
//         *     (low-contrast) regions. The larger the threshold, the less features are produced by the detector.
//         *
//         *     param edgeThreshold The threshold used to filter out edge-like features. Note that the its meaning
//         *     is different from the contrastThreshold, i.e. the larger the edgeThreshold, the less features are
//         *     filtered out (more features are retained).
//         *
//         *     is captured with a weak camera with soft lenses, you might want to reduce the number.
//         * return automatically generated
//         */
//        public static SIFT create(int nfeatures, int nOctaveLayers, double contrastThreshold, double edgeThreshold)
//        {


//            return SIFT.__fromPtr__(xfeatures2d_SIFT_create_11(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold));


//        }

//        /**
//         * param nfeatures The number of best features to retain. The features are ranked by their scores
//         *     (measured in SIFT algorithm as the local contrast)
//         *
//         *     param nOctaveLayers The number of layers in each octave. 3 is the value used in D. Lowe paper. The
//         *     number of octaves is computed automatically from the image resolution.
//         *
//         *     param contrastThreshold The contrast threshold used to filter out weak features in semi-uniform
//         *     (low-contrast) regions. The larger the threshold, the less features are produced by the detector.
//         *
//         *     is different from the contrastThreshold, i.e. the larger the edgeThreshold, the less features are
//         *     filtered out (more features are retained).
//         *
//         *     is captured with a weak camera with soft lenses, you might want to reduce the number.
//         * return automatically generated
//         */
//        public static SIFT create(int nfeatures, int nOctaveLayers, double contrastThreshold)
//        {


//            return SIFT.__fromPtr__(xfeatures2d_SIFT_create_12(nfeatures, nOctaveLayers, contrastThreshold));


//        }

//        /**
//         * param nfeatures The number of best features to retain. The features are ranked by their scores
//         *     (measured in SIFT algorithm as the local contrast)
//         *
//         *     param nOctaveLayers The number of layers in each octave. 3 is the value used in D. Lowe paper. The
//         *     number of octaves is computed automatically from the image resolution.
//         *
//         *     (low-contrast) regions. The larger the threshold, the less features are produced by the detector.
//         *
//         *     is different from the contrastThreshold, i.e. the larger the edgeThreshold, the less features are
//         *     filtered out (more features are retained).
//         *
//         *     is captured with a weak camera with soft lenses, you might want to reduce the number.
//         * return automatically generated
//         */
//        public static SIFT create(int nfeatures, int nOctaveLayers)
//        {


//            return SIFT.__fromPtr__(xfeatures2d_SIFT_create_13(nfeatures, nOctaveLayers));


//        }

//        /**
//         * param nfeatures The number of best features to retain. The features are ranked by their scores
//         *     (measured in SIFT algorithm as the local contrast)
//         *
//         *     number of octaves is computed automatically from the image resolution.
//         *
//         *     (low-contrast) regions. The larger the threshold, the less features are produced by the detector.
//         *
//         *     is different from the contrastThreshold, i.e. the larger the edgeThreshold, the less features are
//         *     filtered out (more features are retained).
//         *
//         *     is captured with a weak camera with soft lenses, you might want to reduce the number.
//         * return automatically generated
//         */
//        public static SIFT create(int nfeatures)
//        {


//            return SIFT.__fromPtr__(xfeatures2d_SIFT_create_14(nfeatures));


//        }

//        /**
//         *     (measured in SIFT algorithm as the local contrast)
//         *
//         *     number of octaves is computed automatically from the image resolution.
//         *
//         *     (low-contrast) regions. The larger the threshold, the less features are produced by the detector.
//         *
//         *     is different from the contrastThreshold, i.e. the larger the edgeThreshold, the less features are
//         *     filtered out (more features are retained).
//         *
//         *     is captured with a weak camera with soft lenses, you might want to reduce the number.
//         * return automatically generated
//         */
//        public static SIFT create()
//        {


//            return SIFT.__fromPtr__(xfeatures2d_SIFT_create_15());


//        }


//#if (UNITY_IOS || UNITY_WEBGL) && !UNITY_EDITOR
//        const string LIBNAME = "__Internal";
//#else
//        const string LIBNAME = "opencvforunity";
//#endif



//        // C++: static Ptr_SIFT cv::xfeatures2d::SIFT::create(int nfeatures = 0, int nOctaveLayers = 3, double contrastThreshold = 0.04, double edgeThreshold = 10, double sigma = 1.6)
//        [DllImport(LIBNAME)]
//        private static extern IntPtr xfeatures2d_SIFT_create_10(int nfeatures, int nOctaveLayers, double contrastThreshold, double edgeThreshold, double sigma);
//        [DllImport(LIBNAME)]
//        private static extern IntPtr xfeatures2d_SIFT_create_11(int nfeatures, int nOctaveLayers, double contrastThreshold, double edgeThreshold);
//        [DllImport(LIBNAME)]
//        private static extern IntPtr xfeatures2d_SIFT_create_12(int nfeatures, int nOctaveLayers, double contrastThreshold);
//        [DllImport(LIBNAME)]
//        private static extern IntPtr xfeatures2d_SIFT_create_13(int nfeatures, int nOctaveLayers);
//        [DllImport(LIBNAME)]
//        private static extern IntPtr xfeatures2d_SIFT_create_14(int nfeatures);
//        [DllImport(LIBNAME)]
//        private static extern IntPtr xfeatures2d_SIFT_create_15();

//        // native support for java finalize()
//        [DllImport(LIBNAME)]
//        private static extern void xfeatures2d_SIFT_delete(IntPtr nativeObj);

//    }
//}
