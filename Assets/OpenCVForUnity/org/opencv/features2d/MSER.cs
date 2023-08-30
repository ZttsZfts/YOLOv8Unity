
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.UtilsModule;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace OpenCVForUnity.Features2dModule
{

    // C++: class MSER
    /**
     * Maximally stable extremal region extractor
     *
     * The class encapsulates all the parameters of the %MSER extraction algorithm (see [wiki
     * article](http://en.wikipedia.org/wiki/Maximally_stable_extremal_regions)).
     *
     * <ul>
     *   <li>
     *  there are two different implementation of %MSER: one for grey image, one for color image
     *   </li>
     * </ul>
     *
     * <ul>
     *   <li>
     *  the grey image algorithm is taken from: CITE: nister2008linear ;  the paper claims to be faster
     * than union-find method; it actually get 1.5~2m/s on my centrino L7200 1.2GHz laptop.
     *   </li>
     * </ul>
     *
     * <ul>
     *   <li>
     *  the color image algorithm is taken from: CITE: forssen2007maximally ; it should be much slower
     * than grey image method ( 3~4 times ); the chi_table.h file is taken directly from paper's source
     * code which is distributed under GPL.
     *   </li>
     * </ul>
     *
     * <ul>
     *   <li>
     *  (Python) A complete example showing the use of the %MSER detector can be found at samples/python/mser.py
     *   </li>
     * </ul>
     */

    public class MSER : Feature2D
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
                        features2d_MSER_delete(nativeObj);
                    nativeObj = IntPtr.Zero;
                }
            }
            finally
            {
                base.Dispose(disposing);
            }

        }

        protected internal MSER(IntPtr addr) : base(addr) { }

        // internal usage only
        public static new MSER __fromPtr__(IntPtr addr) { return new MSER(addr); }

        //
        // C++: static Ptr_MSER cv::MSER::create(int _delta = 5, int _min_area = 60, int _max_area = 14400, double _max_variation = 0.25, double _min_diversity = .2, int _max_evolution = 200, double _area_threshold = 1.01, double _min_margin = 0.003, int _edge_blur_size = 5)
        //

        /**
         * Full consturctor for %MSER detector
         *
         *     param _delta it compares \((size_{i}-size_{i-delta})/size_{i-delta}\)
         *     param _min_area prune the area which smaller than minArea
         *     param _max_area prune the area which bigger than maxArea
         *     param _max_variation prune the area have similar size to its children
         *     param _min_diversity for color image, trace back to cut off mser with diversity less than min_diversity
         *     param _max_evolution  for color image, the evolution steps
         *     param _area_threshold for color image, the area threshold to cause re-initialize
         *     param _min_margin for color image, ignore too small margin
         *     param _edge_blur_size for color image, the aperture size for edge blur
         * return automatically generated
         */
        public static MSER create(int _delta, int _min_area, int _max_area, double _max_variation, double _min_diversity, int _max_evolution, double _area_threshold, double _min_margin, int _edge_blur_size)
        {


            return MSER.__fromPtr__(features2d_MSER_create_10(_delta, _min_area, _max_area, _max_variation, _min_diversity, _max_evolution, _area_threshold, _min_margin, _edge_blur_size));


        }

        /**
         * Full consturctor for %MSER detector
         *
         *     param _delta it compares \((size_{i}-size_{i-delta})/size_{i-delta}\)
         *     param _min_area prune the area which smaller than minArea
         *     param _max_area prune the area which bigger than maxArea
         *     param _max_variation prune the area have similar size to its children
         *     param _min_diversity for color image, trace back to cut off mser with diversity less than min_diversity
         *     param _max_evolution  for color image, the evolution steps
         *     param _area_threshold for color image, the area threshold to cause re-initialize
         *     param _min_margin for color image, ignore too small margin
         * return automatically generated
         */
        public static MSER create(int _delta, int _min_area, int _max_area, double _max_variation, double _min_diversity, int _max_evolution, double _area_threshold, double _min_margin)
        {


            return MSER.__fromPtr__(features2d_MSER_create_11(_delta, _min_area, _max_area, _max_variation, _min_diversity, _max_evolution, _area_threshold, _min_margin));


        }

        /**
         * Full consturctor for %MSER detector
         *
         *     param _delta it compares \((size_{i}-size_{i-delta})/size_{i-delta}\)
         *     param _min_area prune the area which smaller than minArea
         *     param _max_area prune the area which bigger than maxArea
         *     param _max_variation prune the area have similar size to its children
         *     param _min_diversity for color image, trace back to cut off mser with diversity less than min_diversity
         *     param _max_evolution  for color image, the evolution steps
         *     param _area_threshold for color image, the area threshold to cause re-initialize
         * return automatically generated
         */
        public static MSER create(int _delta, int _min_area, int _max_area, double _max_variation, double _min_diversity, int _max_evolution, double _area_threshold)
        {


            return MSER.__fromPtr__(features2d_MSER_create_12(_delta, _min_area, _max_area, _max_variation, _min_diversity, _max_evolution, _area_threshold));


        }

        /**
         * Full consturctor for %MSER detector
         *
         *     param _delta it compares \((size_{i}-size_{i-delta})/size_{i-delta}\)
         *     param _min_area prune the area which smaller than minArea
         *     param _max_area prune the area which bigger than maxArea
         *     param _max_variation prune the area have similar size to its children
         *     param _min_diversity for color image, trace back to cut off mser with diversity less than min_diversity
         *     param _max_evolution  for color image, the evolution steps
         * return automatically generated
         */
        public static MSER create(int _delta, int _min_area, int _max_area, double _max_variation, double _min_diversity, int _max_evolution)
        {


            return MSER.__fromPtr__(features2d_MSER_create_13(_delta, _min_area, _max_area, _max_variation, _min_diversity, _max_evolution));


        }

        /**
         * Full consturctor for %MSER detector
         *
         *     param _delta it compares \((size_{i}-size_{i-delta})/size_{i-delta}\)
         *     param _min_area prune the area which smaller than minArea
         *     param _max_area prune the area which bigger than maxArea
         *     param _max_variation prune the area have similar size to its children
         *     param _min_diversity for color image, trace back to cut off mser with diversity less than min_diversity
         * return automatically generated
         */
        public static MSER create(int _delta, int _min_area, int _max_area, double _max_variation, double _min_diversity)
        {


            return MSER.__fromPtr__(features2d_MSER_create_14(_delta, _min_area, _max_area, _max_variation, _min_diversity));


        }

        /**
         * Full consturctor for %MSER detector
         *
         *     param _delta it compares \((size_{i}-size_{i-delta})/size_{i-delta}\)
         *     param _min_area prune the area which smaller than minArea
         *     param _max_area prune the area which bigger than maxArea
         *     param _max_variation prune the area have similar size to its children
         * return automatically generated
         */
        public static MSER create(int _delta, int _min_area, int _max_area, double _max_variation)
        {


            return MSER.__fromPtr__(features2d_MSER_create_15(_delta, _min_area, _max_area, _max_variation));


        }

        /**
         * Full consturctor for %MSER detector
         *
         *     param _delta it compares \((size_{i}-size_{i-delta})/size_{i-delta}\)
         *     param _min_area prune the area which smaller than minArea
         *     param _max_area prune the area which bigger than maxArea
         * return automatically generated
         */
        public static MSER create(int _delta, int _min_area, int _max_area)
        {


            return MSER.__fromPtr__(features2d_MSER_create_16(_delta, _min_area, _max_area));


        }

        /**
         * Full consturctor for %MSER detector
         *
         *     param _delta it compares \((size_{i}-size_{i-delta})/size_{i-delta}\)
         *     param _min_area prune the area which smaller than minArea
         * return automatically generated
         */
        public static MSER create(int _delta, int _min_area)
        {


            return MSER.__fromPtr__(features2d_MSER_create_17(_delta, _min_area));


        }

        /**
         * Full consturctor for %MSER detector
         *
         *     param _delta it compares \((size_{i}-size_{i-delta})/size_{i-delta}\)
         * return automatically generated
         */
        public static MSER create(int _delta)
        {


            return MSER.__fromPtr__(features2d_MSER_create_18(_delta));


        }

        /**
         * Full consturctor for %MSER detector
         *
         * return automatically generated
         */
        public static MSER create()
        {


            return MSER.__fromPtr__(features2d_MSER_create_19());


        }


        //
        // C++:  String cv::MSER::getDefaultName()
        //

        public override string getDefaultName()
        {
            ThrowIfDisposed();

            string retVal = Marshal.PtrToStringAnsi(features2d_MSER_getDefaultName_10(nativeObj));

            return retVal;
        }


        //
        // C++:  bool cv::MSER::getPass2Only()
        //

        public bool getPass2Only()
        {
            ThrowIfDisposed();

            return features2d_MSER_getPass2Only_10(nativeObj);


        }


        //
        // C++:  int cv::MSER::getDelta()
        //

        public int getDelta()
        {
            ThrowIfDisposed();

            return features2d_MSER_getDelta_10(nativeObj);


        }


        //
        // C++:  int cv::MSER::getMaxArea()
        //

        public int getMaxArea()
        {
            ThrowIfDisposed();

            return features2d_MSER_getMaxArea_10(nativeObj);


        }


        //
        // C++:  int cv::MSER::getMinArea()
        //

        public int getMinArea()
        {
            ThrowIfDisposed();

            return features2d_MSER_getMinArea_10(nativeObj);


        }


        //
        // C++:  void cv::MSER::detectRegions(Mat image, vector_vector_Point& msers, vector_Rect& bboxes)
        //

        /**
         * Detect %MSER regions
         *
         *     param image input image (8UC1, 8UC3 or 8UC4, must be greater or equal than 3x3)
         *     param msers resulting list of point sets
         *     param bboxes resulting bounding boxes
         */
        public void detectRegions(Mat image, List<MatOfPoint> msers, MatOfRect bboxes)
        {
            ThrowIfDisposed();
            if (image != null) image.ThrowIfDisposed();
            if (bboxes != null) bboxes.ThrowIfDisposed();
            Mat msers_mat = new Mat();
            Mat bboxes_mat = bboxes;
            features2d_MSER_detectRegions_10(nativeObj, image.nativeObj, msers_mat.nativeObj, bboxes_mat.nativeObj);
            Converters.Mat_to_vector_vector_Point(msers_mat, msers);
            msers_mat.release();

        }


        //
        // C++:  void cv::MSER::setDelta(int delta)
        //

        public void setDelta(int delta)
        {
            ThrowIfDisposed();

            features2d_MSER_setDelta_10(nativeObj, delta);


        }


        //
        // C++:  void cv::MSER::setMaxArea(int maxArea)
        //

        public void setMaxArea(int maxArea)
        {
            ThrowIfDisposed();

            features2d_MSER_setMaxArea_10(nativeObj, maxArea);


        }


        //
        // C++:  void cv::MSER::setMinArea(int minArea)
        //

        public void setMinArea(int minArea)
        {
            ThrowIfDisposed();

            features2d_MSER_setMinArea_10(nativeObj, minArea);


        }


        //
        // C++:  void cv::MSER::setPass2Only(bool f)
        //

        public void setPass2Only(bool f)
        {
            ThrowIfDisposed();

            features2d_MSER_setPass2Only_10(nativeObj, f);


        }


#if (UNITY_IOS || UNITY_WEBGL) && !UNITY_EDITOR
        const string LIBNAME = "__Internal";
#else
        const string LIBNAME = "opencvforunity";
#endif



        // C++: static Ptr_MSER cv::MSER::create(int _delta = 5, int _min_area = 60, int _max_area = 14400, double _max_variation = 0.25, double _min_diversity = .2, int _max_evolution = 200, double _area_threshold = 1.01, double _min_margin = 0.003, int _edge_blur_size = 5)
        [DllImport(LIBNAME)]
        private static extern IntPtr features2d_MSER_create_10(int _delta, int _min_area, int _max_area, double _max_variation, double _min_diversity, int _max_evolution, double _area_threshold, double _min_margin, int _edge_blur_size);
        [DllImport(LIBNAME)]
        private static extern IntPtr features2d_MSER_create_11(int _delta, int _min_area, int _max_area, double _max_variation, double _min_diversity, int _max_evolution, double _area_threshold, double _min_margin);
        [DllImport(LIBNAME)]
        private static extern IntPtr features2d_MSER_create_12(int _delta, int _min_area, int _max_area, double _max_variation, double _min_diversity, int _max_evolution, double _area_threshold);
        [DllImport(LIBNAME)]
        private static extern IntPtr features2d_MSER_create_13(int _delta, int _min_area, int _max_area, double _max_variation, double _min_diversity, int _max_evolution);
        [DllImport(LIBNAME)]
        private static extern IntPtr features2d_MSER_create_14(int _delta, int _min_area, int _max_area, double _max_variation, double _min_diversity);
        [DllImport(LIBNAME)]
        private static extern IntPtr features2d_MSER_create_15(int _delta, int _min_area, int _max_area, double _max_variation);
        [DllImport(LIBNAME)]
        private static extern IntPtr features2d_MSER_create_16(int _delta, int _min_area, int _max_area);
        [DllImport(LIBNAME)]
        private static extern IntPtr features2d_MSER_create_17(int _delta, int _min_area);
        [DllImport(LIBNAME)]
        private static extern IntPtr features2d_MSER_create_18(int _delta);
        [DllImport(LIBNAME)]
        private static extern IntPtr features2d_MSER_create_19();

        // C++:  String cv::MSER::getDefaultName()
        [DllImport(LIBNAME)]
        private static extern IntPtr features2d_MSER_getDefaultName_10(IntPtr nativeObj);

        // C++:  bool cv::MSER::getPass2Only()
        [DllImport(LIBNAME)]
        [return: MarshalAs(UnmanagedType.U1)]
        private static extern bool features2d_MSER_getPass2Only_10(IntPtr nativeObj);

        // C++:  int cv::MSER::getDelta()
        [DllImport(LIBNAME)]
        private static extern int features2d_MSER_getDelta_10(IntPtr nativeObj);

        // C++:  int cv::MSER::getMaxArea()
        [DllImport(LIBNAME)]
        private static extern int features2d_MSER_getMaxArea_10(IntPtr nativeObj);

        // C++:  int cv::MSER::getMinArea()
        [DllImport(LIBNAME)]
        private static extern int features2d_MSER_getMinArea_10(IntPtr nativeObj);

        // C++:  void cv::MSER::detectRegions(Mat image, vector_vector_Point& msers, vector_Rect& bboxes)
        [DllImport(LIBNAME)]
        private static extern void features2d_MSER_detectRegions_10(IntPtr nativeObj, IntPtr image_nativeObj, IntPtr msers_mat_nativeObj, IntPtr bboxes_mat_nativeObj);

        // C++:  void cv::MSER::setDelta(int delta)
        [DllImport(LIBNAME)]
        private static extern void features2d_MSER_setDelta_10(IntPtr nativeObj, int delta);

        // C++:  void cv::MSER::setMaxArea(int maxArea)
        [DllImport(LIBNAME)]
        private static extern void features2d_MSER_setMaxArea_10(IntPtr nativeObj, int maxArea);

        // C++:  void cv::MSER::setMinArea(int minArea)
        [DllImport(LIBNAME)]
        private static extern void features2d_MSER_setMinArea_10(IntPtr nativeObj, int minArea);

        // C++:  void cv::MSER::setPass2Only(bool f)
        [DllImport(LIBNAME)]
        private static extern void features2d_MSER_setPass2Only_10(IntPtr nativeObj, [MarshalAs(UnmanagedType.U1)] bool f);

        // native support for java finalize()
        [DllImport(LIBNAME)]
        private static extern void features2d_MSER_delete(IntPtr nativeObj);

    }
}
