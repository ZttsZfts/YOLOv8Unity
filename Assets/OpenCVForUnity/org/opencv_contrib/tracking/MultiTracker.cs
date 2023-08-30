
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.UtilsModule;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace OpenCVForUnity.TrackingModule
{

    // C++: class MultiTracker
    /**
     * This class is used to track multiple objects using the specified tracker algorithm.
     *
     * The %MultiTracker is naive implementation of multiple object tracking.
     * It process the tracked objects independently without any optimization accross the tracked objects.
     */

    public class MultiTracker : Algorithm
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
                        tracking_MultiTracker_delete(nativeObj);
                    nativeObj = IntPtr.Zero;
                }
            }
            finally
            {
                base.Dispose(disposing);
            }

        }

        protected internal MultiTracker(IntPtr addr) : base(addr) { }

        // internal usage only
        public static new MultiTracker __fromPtr__(IntPtr addr) { return new MultiTracker(addr); }

        //
        // C++:   cv::MultiTracker::MultiTracker()
        //

        /**
         * Constructor.
         */
        public MultiTracker() :
            base(tracking_MultiTracker_MultiTracker_10())
        {



        }


        //
        // C++: static Ptr_MultiTracker cv::MultiTracker::create()
        //

        /**
         * Returns a pointer to a new instance of MultiTracker
         * return automatically generated
         */
        public static MultiTracker create()
        {


            return MultiTracker.__fromPtr__(tracking_MultiTracker_create_10());


        }


        //
        // C++:  bool cv::MultiTracker::add(Ptr_Tracker newTracker, Mat image, Rect2d boundingBox)
        //

        /**
         * Add a new object to be tracked.
         *
         * param newTracker tracking algorithm to be used
         * param image input image
         * param boundingBox a rectangle represents ROI of the tracked object
         * return automatically generated
         */
        public bool add(Tracker newTracker, Mat image, Rect2d boundingBox)
        {
            ThrowIfDisposed();
            if (newTracker != null) newTracker.ThrowIfDisposed();
            if (image != null) image.ThrowIfDisposed();

            return tracking_MultiTracker_add_10(nativeObj, newTracker.getNativeObjAddr(), image.nativeObj, boundingBox.x, boundingBox.y, boundingBox.width, boundingBox.height);


        }


        //
        // C++:  bool cv::MultiTracker::update(Mat image, vector_Rect2d& boundingBox)
        //

        /**
         * Update the current tracking status.
         * param image input image
         * param boundingBox the tracking result, represent a list of ROIs of the tracked objects.
         * return automatically generated
         */
        public bool update(Mat image, MatOfRect2d boundingBox)
        {
            ThrowIfDisposed();
            if (image != null) image.ThrowIfDisposed();
            if (boundingBox != null) boundingBox.ThrowIfDisposed();
            Mat boundingBox_mat = boundingBox;
            return tracking_MultiTracker_update_10(nativeObj, image.nativeObj, boundingBox_mat.nativeObj);


        }


        //
        // C++:  vector_Rect2d cv::MultiTracker::getObjects()
        //

        /**
         * Returns a reference to a storage for the tracked objects, each object corresponds to one tracker algorithm
         * return automatically generated
         */
        public MatOfRect2d getObjects()
        {
            ThrowIfDisposed();

            return MatOfRect2d.fromNativeAddr(tracking_MultiTracker_getObjects_10(nativeObj));


        }


#if (UNITY_IOS || UNITY_WEBGL) && !UNITY_EDITOR
        const string LIBNAME = "__Internal";
#else
        const string LIBNAME = "opencvforunity";
#endif



        // C++:   cv::MultiTracker::MultiTracker()
        [DllImport(LIBNAME)]
        private static extern IntPtr tracking_MultiTracker_MultiTracker_10();

        // C++: static Ptr_MultiTracker cv::MultiTracker::create()
        [DllImport(LIBNAME)]
        private static extern IntPtr tracking_MultiTracker_create_10();

        // C++:  bool cv::MultiTracker::add(Ptr_Tracker newTracker, Mat image, Rect2d boundingBox)
        [DllImport(LIBNAME)]
        [return: MarshalAs(UnmanagedType.U1)]
        private static extern bool tracking_MultiTracker_add_10(IntPtr nativeObj, IntPtr newTracker_nativeObj, IntPtr image_nativeObj, double boundingBox_x, double boundingBox_y, double boundingBox_width, double boundingBox_height);

        // C++:  bool cv::MultiTracker::update(Mat image, vector_Rect2d& boundingBox)
        [DllImport(LIBNAME)]
        [return: MarshalAs(UnmanagedType.U1)]
        private static extern bool tracking_MultiTracker_update_10(IntPtr nativeObj, IntPtr image_nativeObj, IntPtr boundingBox_mat_nativeObj);

        // C++:  vector_Rect2d cv::MultiTracker::getObjects()
        [DllImport(LIBNAME)]
        private static extern IntPtr tracking_MultiTracker_getObjects_10(IntPtr nativeObj);

        // native support for java finalize()
        [DllImport(LIBNAME)]
        private static extern void tracking_MultiTracker_delete(IntPtr nativeObj);

    }
}
