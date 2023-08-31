using System;
using System.Collections;
using System.Collections.Generic;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.UnityUtils;
using Unity.Barracuda;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.Serialization;
using UnityEngine.UI;
using Tensor = Unity.Barracuda.Tensor;

public class PlayCamera : MonoBehaviour
{
    public NNModel yoloModel;
    public RawImage rawImage;
    public float threshold = 0.7f;

    public Transform wireframeCarrier;
    public Transform wireframe;

    private WebCamTexture _webCamTexture;
    private IWorker _worker;

    private void Start()
    {
        _worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, ModelLoader.Load(yoloModel));
        var webCamDevices = WebCamTexture.devices;
        _webCamTexture = new WebCamTexture(webCamDevices[0].name, 1024, 768, 30);
        _webCamTexture.Play();
        rawImage.texture = _webCamTexture;

        StartCoroutine(Illation(AfterTreatment));
    }

    private void AfterTreatment(List<YoloResult> yoloResults)
    {
        for (var i = 0; i < wireframeCarrier.childCount; i++) Destroy(wireframeCarrier.GetChild(i).gameObject);

        foreach (var yoloResult in yoloResults)
        {
            var label = yoloResult.Label;
            var instantiate = Instantiate(wireframe, wireframeCarrier);
            instantiate.GetComponentsInChildren<Text>()[0].text = label.ToString();
            instantiate.GetComponent<RectTransform>().sizeDelta =
                new Vector2(
                    yoloResult.Rect.Width * (1024f / 640f) *
                    (wireframeCarrier.GetComponent<RectTransform>().rect.width /
                     1024f),
                    yoloResult.Rect.Height * (768f / 640f) *
                    (wireframeCarrier.GetComponent<RectTransform>().rect.height /
                     768f)
                );
            instantiate.GetComponent<RectTransform>().position =
                new Vector3(
                    yoloResult.Rect.X * (1024f / 640f) *
                    (wireframeCarrier.GetComponent<RectTransform>().rect.width / 1024f),
                    wireframeCarrier.GetComponent<RectTransform>().rect.height - yoloResult.Rect.Y * (768f / 640f) *
                    (wireframeCarrier.GetComponent<RectTransform>().rect.height /
                     768f)
                );
        }
    }


    private IEnumerator Illation(UnityAction<List<YoloResult>> process)
    {
        while (true)
        {
            using var cameraInput = new Mat(_webCamTexture.height, _webCamTexture.width, CvType.CV_8UC3);
            Utils.webCamTextureToMat(_webCamTexture, cameraInput);
            using var cameraInputZoom = new Mat(640, 640, CvType.CV_8UC3);
            Imgproc.resize(cameraInput, cameraInputZoom, new Size(cameraInputZoom.cols(), cameraInputZoom.rows()));
            var cameraInputTexture2D = Mat2Texture2D(cameraInputZoom);
            yield return new WaitForEndOfFrame();
            using var inputTensor = new Tensor(cameraInputTexture2D, 3);
            Destroy(cameraInputTexture2D);
            using var outputTensor = _worker.Execute(inputTensor).PeekOutput();

            var yoloResult = new List<YoloResult>();
            var labelNumber = Enum.GetNames(typeof(YoloLabels)).Length;
            for (var i = 0; i < outputTensor.width; i++)
            {
                var yoloItem = new YoloResult();
                yoloItem.Rect = new YoloResultRect();
                yoloItem.Rect.X = outputTensor[0, 0, i, 0];
                yoloItem.Rect.Y = outputTensor[0, 0, i, 1];
                yoloItem.Rect.Width = outputTensor[0, 0, i, 2];
                yoloItem.Rect.Height = outputTensor[0, 0, i, 3];
                var confidenceMax = 0f;
                var confidenceLabel = YoloLabels.Mark1;
                for (var j = 0; j < labelNumber; j++)
                {
                    var confidence = outputTensor[0, 0, i, j + 4];
                    if (confidence >= confidenceMax)
                    {
                        confidenceMax = confidence;
                        confidenceLabel = (YoloLabels)j;
                    }
                }

                yoloItem.Label = confidenceLabel;
                yoloItem.Confidence = confidenceMax;
                if (confidenceMax >= threshold) yoloResult.Add(yoloItem);
            }

            process(yoloResult);
            yield return new WaitForSeconds(0.2f);
        }
    }

    private Texture2D Mat2Texture2D(Mat mat)
    {
        var texture = new Texture2D(mat.cols(), mat.rows(), TextureFormat.RGB24, false);
        Utils.matToTexture2D(mat, texture);
        return texture;
    }

    public void OnDestroy()
    {
        _webCamTexture.Stop();
        _worker.Dispose();
    }
}

internal enum YoloLabels
{
    Mark1,
    Mark2
}

internal struct YoloResult
{
    public YoloLabels Label;
    public float Confidence;
    public YoloResultRect Rect;
}

internal struct YoloResultRect
{
    public float X;
    public float Y;
    public float Width;
    public float Height;
}