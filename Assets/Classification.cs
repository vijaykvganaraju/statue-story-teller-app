using UnityEngine;
using Unity.Barracuda;
using System.Collections.Generic;
using System.Linq;
using System.Collections;
using UnityEngine.UI;
using UnityEngine.Video;
using System;
//using Microsoft.ML.Transforms;

public class Classification : MonoBehaviour {

	const int IMAGE_SIZE = 224;
	const string INPUT_NAME = "serving_default_keras_layer_input";
	const string OUTPUT_NAME = "StatefulPartitionedCall";


	[Header("Model Stuff")]
	public NNModel modelFile;
	public TextAsset labelAsset;

	[Header("Scene Stuff")]
	public CameraView CameraView;
	public Preprocess preprocess;
	public Text uiText;
	public RawImage TheImage;
	public Texture[] myTextures = new Texture[2];
	public VideoPlayer video;

	private int currentItem = 0;
	private float max_prob;
	private string class_name;

	//string[] labels;
	string[] labels;
	IWorker worker;

	void Start() {
        var model = ModelLoader.Load(modelFile);
        worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, model);
				//video = GetComponent<VideoPlayer>();
				video.Pause();
        LoadLabels();
				TheImage.texture = myTextures[0];
	}

	void LoadLabels() {
		//get only items in quotes
		var stringArray = labelAsset.text.Split('"').Where((item, index) => index % 2 != 0);
		//Debug.Log(labelAsset.text);
		//get every other item
		labels = stringArray.Where((x, i) => i % 2 != 0).ToArray();
		//foreach (var item in labels) {
      //      Debug.Log(item);
        //}
	}

	void Update() {

		WebCamTexture webCamTexture = CameraView.GetCamImage();

		if (webCamTexture.didUpdateThisFrame && webCamTexture.width > 100) {
			preprocess.ScaleAndCropImage(webCamTexture, IMAGE_SIZE, RunModel);
		}
			if (class_name == "WILLIAMJOSEPHCHAMINADE" && max_prob > 0.85)
			{
				video.Play();
				TheImage.texture = myTextures[1];


			}
			//else if (uiText.text == "FLAMETANAGER" && currentItem!= 2)
			//{
				// TheImage.texture = myTextures[2];
				 //currentItem = 2;
			//}

	}

	void RunModel(byte[] pixels) {
		StartCoroutine(RunModelRoutine(pixels));
	}

	IEnumerator RunModelRoutine(byte[] pixels) {

		Tensor tensor = TransformInput(pixels);

		var inputs = new Dictionary<string, Tensor> {
			{ INPUT_NAME, tensor }
		};

		worker.Execute(inputs);
		Tensor outputTensor = worker.PeekOutput(OUTPUT_NAME);

		//get largest output
		List<float> temp = outputTensor.ToReadOnlyArray().ToList();
		float max = temp.Max();
		max_prob = max;
		int index = temp.IndexOf(max);
		class_name = labels[index];

        //set UI text
        uiText.text = max_prob.ToString("0.0") + labels[index];
				//if (uiText.text == "bathing Cap," && currentItem!= 1)
        //{
           //TheImage.texture = myTextures[1];
					// currentItem = 1;
        //}
				//else if (currentItem!=1){
					// TheImage.texture = myTextures[0];
				//}

				//rawImage.texture = webcamTexture;

        //dispose tensors
        tensor.Dispose();
		outputTensor.Dispose();
		yield return null;
	}

	//transform from 0-255 to -1 to 1
	Tensor TransformInput(byte[] pixels){
		float[] transformedPixels = new float[pixels.Length];

		//Debug.Log(pixels[10]);

		for (int i = 0; i < pixels.Length; i++){
			transformedPixels[i] = (pixels[i] - 127f) / 128f;
			//transformedPixels[i] = pixels[i];
			//transformedPixels[i] = (pixels[i]) / 255f;
			//if(i%3==0){
			//transformedPixels[i] = ((transformedPixels[i]) - 0.485f) / 0.229f;}
			//if(i%3==1){
			//transformedPixels[i] = ((transformedPixels[i]) - 0.456f) / 0.224f;}
			//if(i%3==2){
			//transformedPixels[i] = ((transformedPixels[i]) - 0.406f) / 0.225f;}
			//var color = pixels[i];

			//transformedPixels[i * 3 + 0] = (color.r - 0.485f) / 0.229f;
			//transformedPixels[i * 3 + 1] = (color.g - 0.456f) / 0.224f;
			//transformedPixels[i * 3 + 2] = (color.b - 0.406f) / 0.225f;

		}
		return new Tensor(1, IMAGE_SIZE, IMAGE_SIZE, 3, transformedPixels);
	}
}
