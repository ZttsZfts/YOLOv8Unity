
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.UtilsModule;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace OpenCVForUnity.XimgprocModule
{

    // C++: class FastLineDetector
    /**
     * Class implementing the FLD (Fast Line Detector) algorithm described
     * in CITE: Lee14 .
     */

    public class FastLineDetector : Algorithm
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
                        ximgproc_FastLineDetector_delete(nativeObj);
                    nativeObj = IntPtr.Zero;
                }
            }
            finally
            {
                base.Dispose(disposing);
            }

        }

        protected internal FastLineDetector(IntPtr addr) : base(addr) { }

        // internal usage only
        public static new FastLineDetector __fromPtr__(IntPtr addr) { return new FastLineDetector(addr); }

        //
        // C++:  void cv::ximgproc::FastLineDetector::detect(Mat _image, Mat& _lines)
        //

        /**
         * Finds lines in the input image.
         *       This is the output of the default parameters of the algorithm on the above
         *       shown image.
         *
         *       ![image](pics/corridor_fld.jpg)
         *
         *       param _image A grayscale (CV_8UC1) input image. If only a roi needs to be
         *       selected, use: `fld_ptr-&gt;detect(image(roi), lines, ...);
         *       lines += Scalar(roi.x, roi.y, roi.x, roi.y);`
         *       param _lines A vector of Vec4f elements specifying the beginning
         *       and ending point of a line.  Where Vec4f is (x1, y1, x2, y2), point
         *       1 is the start, point 2 - end. Returned lines are directed so that the
         *       brighter side is on their left.
         */
        public void detect(Mat _image, Mat _lines)
        {
            ThrowIfDisposed();
            if (_image != null) _image.ThrowIfDisposed();
            if (_lines != null) _lines.ThrowIfDisposed();

            ximgproc_FastLineDetector_detect_10(nativeObj, _image.nativeObj, _lines.nativeObj);


        }


        //
        // C++:  void cv::ximgproc::FastLineDetector::drawSegments(Mat& _image, Mat lines, bool draw_arrow = false)
        //

        /**
         * Draws the line segments on a given image.
         *       param _image The image, where the lines will be drawn. Should be bigger
         *       or equal to the image, where the lines were found.
         *       param lines A vector of the lines that needed to be drawn.
         *       param draw_arrow If true, arrow heads will be drawn.
         */
        public void drawSegments(Mat _image, Mat lines, bool draw_arrow)
        {
            ThrowIfDisposed();
            if (_image != null) _image.ThrowIfDisposed();
            if (lines != null) lines.ThrowIfDisposed();

            ximgproc_FastLineDetector_drawSegments_10(nativeObj, _image.nativeObj, lines.nativeObj, draw_arrow);


        }

        /**
         * Draws the line segments on a given image.
         *       param _image The image, where the lines will be drawn. Should be bigger
         *       or equal to the image, where the lines were found.
         *       param lines A vector of the lines that needed to be drawn.
         */
        public void drawSegments(Mat _image, Mat lines)
        {
            ThrowIfDisposed();
            if (_image != null) _image.ThrowIfDisposed();
            if (lines != null) lines.ThrowIfDisposed();

            ximgproc_FastLineDetector_drawSegments_11(nativeObj, _image.nativeObj, lines.nativeObj);


        }


#if (UNITY_IOS || UNITY_WEBGL) && !UNITY_EDITOR
        const string LIBNAME = "__Internal";
#else
        const string LIBNAME = "opencvforunity";
#endif



        // C++:  void cv::ximgproc::FastLineDetector::detect(Mat _image, Mat& _lines)
        [DllImport(LIBNAME)]
        private static extern void ximgproc_FastLineDetector_detect_10(IntPtr nativeObj, IntPtr _image_nativeObj, IntPtr _lines_nativeObj);

        // C++:  void cv::ximgproc::FastLineDetector::drawSegments(Mat& _image, Mat lines, bool draw_arrow = false)
        [DllImport(LIBNAME)]
        private static extern void ximgproc_FastLineDetector_drawSegments_10(IntPtr nativeObj, IntPtr _image_nativeObj, IntPtr lines_nativeObj, [MarshalAs(UnmanagedType.U1)] bool draw_arrow);
        [DllImport(LIBNAME)]
        private static extern void ximgproc_FastLineDetector_drawSegments_11(IntPtr nativeObj, IntPtr _image_nativeObj, IntPtr lines_nativeObj);

        // native support for java finalize()
        [DllImport(LIBNAME)]
        private static extern void ximgproc_FastLineDetector_delete(IntPtr nativeObj);

    }
}
