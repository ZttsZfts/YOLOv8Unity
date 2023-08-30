

using OpenCVForUnity.CoreModule;
using OpenCVForUnity.UtilsModule;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace OpenCVForUnity.ObjdetectModule
{
    // C++: class QRCodeDetector
    /**
     * Groups the object candidate rectangles.
     *     rectList  Input/output vector of rectangles. Output vector includes retained and grouped rectangles. (The Python list is not modified in place.)
     *     weights Input/output vector of weights of rectangles. Output vector includes weights of retained and grouped rectangles. (The Python list is not modified in place.)
     *     groupThreshold Minimum possible number of rectangles minus 1. The threshold is used in a group of rectangles to retain it.
     *     eps Relative difference between sides of the rectangles to merge them into a group.
     */

    public class QRCodeDetector : DisposableOpenCVObject
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
                        objdetect_QRCodeDetector_delete(nativeObj);
                    nativeObj = IntPtr.Zero;
                }
            }
            finally
            {
                base.Dispose(disposing);
            }

        }

        protected internal QRCodeDetector(IntPtr addr) : base(addr) { }


        public IntPtr getNativeObjAddr() { return nativeObj; }

        // internal usage only
        public static QRCodeDetector __fromPtr__(IntPtr addr) { return new QRCodeDetector(addr); }

        //
        // C++:   cv::QRCodeDetector::QRCodeDetector()
        //

        public QRCodeDetector()
        {


            nativeObj = objdetect_QRCodeDetector_QRCodeDetector_10();


        }


        //
        // C++:  bool cv::QRCodeDetector::detect(Mat img, Mat& points)
        //

        /**
         * Detects QR code in image and returns the quadrangle containing the code.
         *      param img grayscale or color (BGR) image containing (or not) QR code.
         *      param points Output vector of vertices of the minimum-area quadrangle containing the code.
         * return automatically generated
         */
        public bool detect(Mat img, Mat points)
        {
            ThrowIfDisposed();
            if (img != null) img.ThrowIfDisposed();
            if (points != null) points.ThrowIfDisposed();

            return objdetect_QRCodeDetector_detect_10(nativeObj, img.nativeObj, points.nativeObj);


        }


        //
        // C++:  string cv::QRCodeDetector::decode(Mat img, Mat points, Mat& straight_qrcode = Mat())
        //

        /**
         * Decodes QR code in image once it's found by the detect() method.
         *      Returns UTF8-encoded output string or empty string if the code cannot be decoded.
         *
         *      param img grayscale or color (BGR) image containing QR code.
         *      param points Quadrangle vertices found by detect() method (or some other algorithm).
         *      param straight_qrcode The optional output image containing rectified and binarized QR code
         * return automatically generated
         */
        public string decode(Mat img, Mat points, Mat straight_qrcode)
        {
            ThrowIfDisposed();
            if (img != null) img.ThrowIfDisposed();
            if (points != null) points.ThrowIfDisposed();
            if (straight_qrcode != null) straight_qrcode.ThrowIfDisposed();

            string retVal = Marshal.PtrToStringAnsi(objdetect_QRCodeDetector_decode_10(nativeObj, img.nativeObj, points.nativeObj, straight_qrcode.nativeObj));

            return retVal;
        }

        /**
         * Decodes QR code in image once it's found by the detect() method.
         *      Returns UTF8-encoded output string or empty string if the code cannot be decoded.
         *
         *      param img grayscale or color (BGR) image containing QR code.
         *      param points Quadrangle vertices found by detect() method (or some other algorithm).
         * return automatically generated
         */
        public string decode(Mat img, Mat points)
        {
            ThrowIfDisposed();
            if (img != null) img.ThrowIfDisposed();
            if (points != null) points.ThrowIfDisposed();

            string retVal = Marshal.PtrToStringAnsi(objdetect_QRCodeDetector_decode_11(nativeObj, img.nativeObj, points.nativeObj));

            return retVal;
        }


        //
        // C++:  string cv::QRCodeDetector::detectAndDecode(Mat img, Mat& points = Mat(), Mat& straight_qrcode = Mat())
        //

        /**
         * Both detects and decodes QR code
         *
         *      param img grayscale or color (BGR) image containing QR code.
         *      param points opiotnal output array of vertices of the found QR code quadrangle. Will be empty if not found.
         *      param straight_qrcode The optional output image containing rectified and binarized QR code
         * return automatically generated
         */
        public string detectAndDecode(Mat img, Mat points, Mat straight_qrcode)
        {
            ThrowIfDisposed();
            if (img != null) img.ThrowIfDisposed();
            if (points != null) points.ThrowIfDisposed();
            if (straight_qrcode != null) straight_qrcode.ThrowIfDisposed();

            string retVal = Marshal.PtrToStringAnsi(objdetect_QRCodeDetector_detectAndDecode_10(nativeObj, img.nativeObj, points.nativeObj, straight_qrcode.nativeObj));

            return retVal;
        }

        /**
         * Both detects and decodes QR code
         *
         *      param img grayscale or color (BGR) image containing QR code.
         *      param points opiotnal output array of vertices of the found QR code quadrangle. Will be empty if not found.
         * return automatically generated
         */
        public string detectAndDecode(Mat img, Mat points)
        {
            ThrowIfDisposed();
            if (img != null) img.ThrowIfDisposed();
            if (points != null) points.ThrowIfDisposed();

            string retVal = Marshal.PtrToStringAnsi(objdetect_QRCodeDetector_detectAndDecode_11(nativeObj, img.nativeObj, points.nativeObj));

            return retVal;
        }

        /**
         * Both detects and decodes QR code
         *
         *      param img grayscale or color (BGR) image containing QR code.
         * return automatically generated
         */
        public string detectAndDecode(Mat img)
        {
            ThrowIfDisposed();
            if (img != null) img.ThrowIfDisposed();

            string retVal = Marshal.PtrToStringAnsi(objdetect_QRCodeDetector_detectAndDecode_12(nativeObj, img.nativeObj));

            return retVal;
        }


        //
        // C++:  void cv::QRCodeDetector::setEpsX(double epsX)
        //

        /**
         * sets the epsilon used during the horizontal scan of QR code stop marker detection.
         *      param epsX Epsilon neighborhood, which allows you to determine the horizontal pattern
         *      of the scheme 1:1:3:1:1 according to QR code standard.
         */
        public void setEpsX(double epsX)
        {
            ThrowIfDisposed();

            objdetect_QRCodeDetector_setEpsX_10(nativeObj, epsX);


        }


        //
        // C++:  void cv::QRCodeDetector::setEpsY(double epsY)
        //

        /**
         * sets the epsilon used during the vertical scan of QR code stop marker detection.
         *      param epsY Epsilon neighborhood, which allows you to determine the vertical pattern
         *      of the scheme 1:1:3:1:1 according to QR code standard.
         */
        public void setEpsY(double epsY)
        {
            ThrowIfDisposed();

            objdetect_QRCodeDetector_setEpsY_10(nativeObj, epsY);


        }


#if (UNITY_IOS || UNITY_WEBGL) && !UNITY_EDITOR
        const string LIBNAME = "__Internal";
#else
        const string LIBNAME = "opencvforunity";
#endif



        // C++:   cv::QRCodeDetector::QRCodeDetector()
        [DllImport(LIBNAME)]
        private static extern IntPtr objdetect_QRCodeDetector_QRCodeDetector_10();

        // C++:  bool cv::QRCodeDetector::detect(Mat img, Mat& points)
        [DllImport(LIBNAME)]
        [return: MarshalAs(UnmanagedType.U1)]
        private static extern bool objdetect_QRCodeDetector_detect_10(IntPtr nativeObj, IntPtr img_nativeObj, IntPtr points_nativeObj);

        // C++:  string cv::QRCodeDetector::decode(Mat img, Mat points, Mat& straight_qrcode = Mat())
        [DllImport(LIBNAME)]
        private static extern IntPtr objdetect_QRCodeDetector_decode_10(IntPtr nativeObj, IntPtr img_nativeObj, IntPtr points_nativeObj, IntPtr straight_qrcode_nativeObj);
        [DllImport(LIBNAME)]
        private static extern IntPtr objdetect_QRCodeDetector_decode_11(IntPtr nativeObj, IntPtr img_nativeObj, IntPtr points_nativeObj);

        // C++:  string cv::QRCodeDetector::detectAndDecode(Mat img, Mat& points = Mat(), Mat& straight_qrcode = Mat())
        [DllImport(LIBNAME)]
        private static extern IntPtr objdetect_QRCodeDetector_detectAndDecode_10(IntPtr nativeObj, IntPtr img_nativeObj, IntPtr points_nativeObj, IntPtr straight_qrcode_nativeObj);
        [DllImport(LIBNAME)]
        private static extern IntPtr objdetect_QRCodeDetector_detectAndDecode_11(IntPtr nativeObj, IntPtr img_nativeObj, IntPtr points_nativeObj);
        [DllImport(LIBNAME)]
        private static extern IntPtr objdetect_QRCodeDetector_detectAndDecode_12(IntPtr nativeObj, IntPtr img_nativeObj);

        // C++:  void cv::QRCodeDetector::setEpsX(double epsX)
        [DllImport(LIBNAME)]
        private static extern void objdetect_QRCodeDetector_setEpsX_10(IntPtr nativeObj, double epsX);

        // C++:  void cv::QRCodeDetector::setEpsY(double epsY)
        [DllImport(LIBNAME)]
        private static extern void objdetect_QRCodeDetector_setEpsY_10(IntPtr nativeObj, double epsY);

        // native support for java finalize()
        [DllImport(LIBNAME)]
        private static extern void objdetect_QRCodeDetector_delete(IntPtr nativeObj);

    }
}
