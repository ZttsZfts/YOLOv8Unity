#if !UNITY_WSA_10_0

using OpenCVForUnity.CoreModule;
using OpenCVForUnity.UtilsModule;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace OpenCVForUnity.TextModule
{

    // C++: class OCRHMMDecoder
    /**
     * OCRHMMDecoder class provides an interface for OCR using Hidden Markov Models.
     *
     * <b>Note:</b>
     * <ul>
     *   <li>
     *       (C++) An example on using OCRHMMDecoder recognition combined with scene text detection can
     *         be found at the webcam_demo sample:
     *         &lt;https://github.com/opencv/opencv_contrib/blob/master/modules/text/samples/webcam_demo.cpp&gt;
     *   </li>
     * </ul>
     */

    public class OCRHMMDecoder : BaseOCR
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
                        text_OCRHMMDecoder_delete(nativeObj);
                    nativeObj = IntPtr.Zero;
                }
            }
            finally
            {
                base.Dispose(disposing);
            }

        }

        protected internal OCRHMMDecoder(IntPtr addr) : base(addr) { }

        // internal usage only
        public static new OCRHMMDecoder __fromPtr__(IntPtr addr) { return new OCRHMMDecoder(addr); }

        //
        // C++: static Ptr_OCRHMMDecoder cv::text::OCRHMMDecoder::create(Ptr_OCRHMMDecoder_ClassifierCallback classifier, String vocabulary, Mat transition_probabilities_table, Mat emission_probabilities_table, int mode = OCR_DECODER_VITERBI)
        //

        // Unknown type 'Ptr_OCRHMMDecoder_ClassifierCallback' (I), skipping the function


        //
        // C++: static Ptr_OCRHMMDecoder cv::text::OCRHMMDecoder::create(String filename, String vocabulary, Mat transition_probabilities_table, Mat emission_probabilities_table, int mode = OCR_DECODER_VITERBI, int classifier = OCR_KNN_CLASSIFIER)
        //

        /**
         * Creates an instance of the OCRHMMDecoder class. Loads and initializes HMMDecoder from the specified path
         *
         *      
         * param filename automatically generated
         * param vocabulary automatically generated
         * param transition_probabilities_table automatically generated
         * param emission_probabilities_table automatically generated
         * param mode automatically generated
         * param classifier automatically generated
         * return automatically generated
         */
        public static OCRHMMDecoder create(string filename, string vocabulary, Mat transition_probabilities_table, Mat emission_probabilities_table, int mode, int classifier)
        {
            if (transition_probabilities_table != null) transition_probabilities_table.ThrowIfDisposed();
            if (emission_probabilities_table != null) emission_probabilities_table.ThrowIfDisposed();

            return OCRHMMDecoder.__fromPtr__(text_OCRHMMDecoder_create_10(filename, vocabulary, transition_probabilities_table.nativeObj, emission_probabilities_table.nativeObj, mode, classifier));


        }

        /**
         * Creates an instance of the OCRHMMDecoder class. Loads and initializes HMMDecoder from the specified path
         *
         *      
         * param filename automatically generated
         * param vocabulary automatically generated
         * param transition_probabilities_table automatically generated
         * param emission_probabilities_table automatically generated
         * param mode automatically generated
         * return automatically generated
         */
        public static OCRHMMDecoder create(string filename, string vocabulary, Mat transition_probabilities_table, Mat emission_probabilities_table, int mode)
        {
            if (transition_probabilities_table != null) transition_probabilities_table.ThrowIfDisposed();
            if (emission_probabilities_table != null) emission_probabilities_table.ThrowIfDisposed();

            return OCRHMMDecoder.__fromPtr__(text_OCRHMMDecoder_create_11(filename, vocabulary, transition_probabilities_table.nativeObj, emission_probabilities_table.nativeObj, mode));


        }

        /**
         * Creates an instance of the OCRHMMDecoder class. Loads and initializes HMMDecoder from the specified path
         *
         *      
         * param filename automatically generated
         * param vocabulary automatically generated
         * param transition_probabilities_table automatically generated
         * param emission_probabilities_table automatically generated
         * return automatically generated
         */
        public static OCRHMMDecoder create(string filename, string vocabulary, Mat transition_probabilities_table, Mat emission_probabilities_table)
        {
            if (transition_probabilities_table != null) transition_probabilities_table.ThrowIfDisposed();
            if (emission_probabilities_table != null) emission_probabilities_table.ThrowIfDisposed();

            return OCRHMMDecoder.__fromPtr__(text_OCRHMMDecoder_create_12(filename, vocabulary, transition_probabilities_table.nativeObj, emission_probabilities_table.nativeObj));


        }


        //
        // C++:  String cv::text::OCRHMMDecoder::run(Mat image, Mat mask, int min_confidence, int component_level = 0)
        //

        public string run(Mat image, Mat mask, int min_confidence, int component_level)
        {
            ThrowIfDisposed();
            if (image != null) image.ThrowIfDisposed();
            if (mask != null) mask.ThrowIfDisposed();

            string retVal = Marshal.PtrToStringAnsi(text_OCRHMMDecoder_run_10(nativeObj, image.nativeObj, mask.nativeObj, min_confidence, component_level));

            return retVal;
        }

        public string run(Mat image, Mat mask, int min_confidence)
        {
            ThrowIfDisposed();
            if (image != null) image.ThrowIfDisposed();
            if (mask != null) mask.ThrowIfDisposed();

            string retVal = Marshal.PtrToStringAnsi(text_OCRHMMDecoder_run_11(nativeObj, image.nativeObj, mask.nativeObj, min_confidence));

            return retVal;
        }


        //
        // C++:  String cv::text::OCRHMMDecoder::run(Mat image, int min_confidence, int component_level = 0)
        //

        /**
         * Recognize text using HMM.
         *
         *     Takes an image and a mask (where each connected component corresponds to a segmented character)
         *     on input and returns recognized text in the output_text parameter. Optionally
         *     provides also the Rects for individual text elements found (e.g. words), and the list of those
         *     text elements with their confidence values.
         *
         *     param image Input image CV_8UC1 or CV_8UC3 with a single text line (or word).
         *
         *
         *     text elements found (e.g. words).
         *
         *     recognition of individual text elements found (e.g. words).
         *
         *     for the recognition of individual text elements found (e.g. words).
         *
         *     param component_level Only OCR_LEVEL_WORD is supported.
         * param min_confidence automatically generated
         * return automatically generated
         */
        public string run(Mat image, int min_confidence, int component_level)
        {
            ThrowIfDisposed();
            if (image != null) image.ThrowIfDisposed();

            string retVal = Marshal.PtrToStringAnsi(text_OCRHMMDecoder_run_12(nativeObj, image.nativeObj, min_confidence, component_level));

            return retVal;
        }

        /**
         * Recognize text using HMM.
         *
         *     Takes an image and a mask (where each connected component corresponds to a segmented character)
         *     on input and returns recognized text in the output_text parameter. Optionally
         *     provides also the Rects for individual text elements found (e.g. words), and the list of those
         *     text elements with their confidence values.
         *
         *     param image Input image CV_8UC1 or CV_8UC3 with a single text line (or word).
         *
         *
         *     text elements found (e.g. words).
         *
         *     recognition of individual text elements found (e.g. words).
         *
         *     for the recognition of individual text elements found (e.g. words).
         *
         * param min_confidence automatically generated
         * return automatically generated
         */
        public string run(Mat image, int min_confidence)
        {
            ThrowIfDisposed();
            if (image != null) image.ThrowIfDisposed();

            string retVal = Marshal.PtrToStringAnsi(text_OCRHMMDecoder_run_13(nativeObj, image.nativeObj, min_confidence));

            return retVal;
        }


#if (UNITY_IOS || UNITY_WEBGL) && !UNITY_EDITOR
        const string LIBNAME = "__Internal";
#else
        const string LIBNAME = "opencvforunity";
#endif



        // C++: static Ptr_OCRHMMDecoder cv::text::OCRHMMDecoder::create(String filename, String vocabulary, Mat transition_probabilities_table, Mat emission_probabilities_table, int mode = OCR_DECODER_VITERBI, int classifier = OCR_KNN_CLASSIFIER)
        [DllImport(LIBNAME)]
        private static extern IntPtr text_OCRHMMDecoder_create_10(string filename, string vocabulary, IntPtr transition_probabilities_table_nativeObj, IntPtr emission_probabilities_table_nativeObj, int mode, int classifier);
        [DllImport(LIBNAME)]
        private static extern IntPtr text_OCRHMMDecoder_create_11(string filename, string vocabulary, IntPtr transition_probabilities_table_nativeObj, IntPtr emission_probabilities_table_nativeObj, int mode);
        [DllImport(LIBNAME)]
        private static extern IntPtr text_OCRHMMDecoder_create_12(string filename, string vocabulary, IntPtr transition_probabilities_table_nativeObj, IntPtr emission_probabilities_table_nativeObj);

        // C++:  String cv::text::OCRHMMDecoder::run(Mat image, Mat mask, int min_confidence, int component_level = 0)
        [DllImport(LIBNAME)]
        private static extern IntPtr text_OCRHMMDecoder_run_10(IntPtr nativeObj, IntPtr image_nativeObj, IntPtr mask_nativeObj, int min_confidence, int component_level);
        [DllImport(LIBNAME)]
        private static extern IntPtr text_OCRHMMDecoder_run_11(IntPtr nativeObj, IntPtr image_nativeObj, IntPtr mask_nativeObj, int min_confidence);

        // C++:  String cv::text::OCRHMMDecoder::run(Mat image, int min_confidence, int component_level = 0)
        [DllImport(LIBNAME)]
        private static extern IntPtr text_OCRHMMDecoder_run_12(IntPtr nativeObj, IntPtr image_nativeObj, int min_confidence, int component_level);
        [DllImport(LIBNAME)]
        private static extern IntPtr text_OCRHMMDecoder_run_13(IntPtr nativeObj, IntPtr image_nativeObj, int min_confidence);

        // native support for java finalize()
        [DllImport(LIBNAME)]
        private static extern void text_OCRHMMDecoder_delete(IntPtr nativeObj);

    }
}
#endif