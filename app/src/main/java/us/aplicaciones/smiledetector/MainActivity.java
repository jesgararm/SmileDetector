package us.aplicaciones.smiledetector;

import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.DataType;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

/**
 * MainActivity es la actividad principal de la aplicación que maneja la captura de fotos,
 * la selección de imágenes de la galería y la detección de sonrisas en las imágenes.
 */
public class MainActivity extends AppCompatActivity {

    private static final int REQUEST_IMAGE_CAPTURE = 1;
    private static final int REQUEST_SELECT_IMAGE = 2;

    private ImageView imageView;
    private TextView textResult;
    private Interpreter tfliteInterpreter;

    /**
     * Método onCreate que se llama cuando se crea la actividad.
     * @param savedInstanceState Estado guardado de la instancia anterior
     */
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = findViewById(R.id.image_view);
        textResult = findViewById(R.id.text_result);

        Button btnTakePhoto = findViewById(R.id.btn_take_photo);
        Button btnSelectGallery = findViewById(R.id.btn_select_gallery);

        // Configura los listeners para los botones de tomar foto y seleccionar de la galería
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            btnTakePhoto.setOnClickListener(v -> capturePhoto());
        }
        btnSelectGallery.setOnClickListener(v -> selectFromGallery());

        // Carga el modelo de detección de sonrisas
        try {
            tfliteInterpreter = new Interpreter(loadModelFile());
            Toast.makeText(this, "Modelo cargado correctamente", Toast.LENGTH_SHORT).show();
        } catch (Exception e) {
            e.printStackTrace();
            Toast.makeText(this, "Error al cargar el modelo", Toast.LENGTH_LONG).show();
        }
    }

    /**
     * Inicia la captura de una foto utilizando la cámara.
     */
    @RequiresApi(api = Build.VERSION_CODES.M)
    @SuppressLint("QueryPermissionsNeeded")
    private void capturePhoto() {
        if (checkSelfPermission(android.Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{android.Manifest.permission.CAMERA}, REQUEST_IMAGE_CAPTURE);
        } else {
            Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
            if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
                startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE);
            } else {
                Toast.makeText(this, "No se pudo abrir la cámara", Toast.LENGTH_SHORT).show();
            }
        }
    }

    /**
     * Inicia la selección de una imagen desde la galería.
     */
    private void selectFromGallery() {
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(intent, REQUEST_SELECT_IMAGE);
    }

    /**
     * Maneja el resultado de las actividades de captura de foto y selección de imagen.
     * @param requestCode Código de solicitud.
     * @param resultCode Código de resultado.
     * @param data Datos del resultado.
     */
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode == RESULT_OK && data != null) {
            Bitmap bitmap = null;
            try {
                if (requestCode == REQUEST_IMAGE_CAPTURE) {
                    if (data.getExtras() != null) {
                        bitmap = (Bitmap) data.getExtras().get("data");
                    }
                } else if (requestCode == REQUEST_SELECT_IMAGE) {
                    if (data.getData() != null) {
                        bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), data.getData());
                    }
                }

                if (bitmap != null) {
                    imageView.setImageBitmap(bitmap);
                    detectSmile(bitmap);
                } else {
                    Toast.makeText(this, "Error: No se obtuvo una imagen válida", Toast.LENGTH_SHORT).show();
                }
            } catch (Exception e) {
                e.printStackTrace();
                Toast.makeText(this, "Error procesando la imagen", Toast.LENGTH_SHORT).show();
            }
        } else {
            Toast.makeText(this, "No se seleccionó ninguna imagen", Toast.LENGTH_SHORT).show();
        }
    }

    /**
     * Detecta una sonrisa en la imagen proporcionada.
     * @param bitmap Imagen en la que se va a detectar la sonrisa.
     */
    @SuppressLint("DefaultLocale")
    private void detectSmile(Bitmap bitmap) {
        try {
            TensorImage tensorImage = preprocessImage(bitmap);

            if (tensorImage == null) {
                Toast.makeText(this, "Error al preprocesar la imagen", Toast.LENGTH_SHORT).show();
                return;
            }

            float[][] output = new float[1][1];
            tfliteInterpreter.run(tensorImage.getBuffer(), output);

            float probability = output[0][0];
            String result = probability > 0.5 ? "Sonrisa detectada!" : "No hay sonrisa.";
            textResult.setText(String.format("%s (Probabilidad: %.2f)", result, probability));
        } catch (Exception e) {
            e.printStackTrace();
            Toast.makeText(this, "Error al realizar la predicción", Toast.LENGTH_SHORT).show();
        }
    }

    /**
     * Preprocesa la imagen para que sea compatible con el modelo de detección de sonrisas.
     * @param bitmap Imagen a preprocesar.
     * @return Imagen preprocesada.
     */
    private TensorImage preprocessImage(Bitmap bitmap) {
        try {
            TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
            tensorImage.load(bitmap);

            // Aplicar las transformaciones necesarias
            ImageProcessor imageProcessor = new ImageProcessor.Builder()
                    .add(new ResizeOp(128, 128, ResizeOp.ResizeMethod.BILINEAR)) // Redimensionar
                    .add(new NormalizeOp(0.0f, 1.0f)) // Normalizar entre 0 y 1
                    .build();

            return imageProcessor.process(tensorImage);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    /**
     * Carga el archivo del modelo de detección de sonrisas.
     * @return MappedByteBuffer con el contenido del archivo del modelo.
     * @throws IOException Si ocurre un error al cargar el archivo.
     */
    private MappedByteBuffer loadModelFile() throws IOException {
        try {
            AssetFileDescriptor fileDescriptor = getAssets().openFd("smile_detection_model.tflite");
            FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        } catch (IOException e) {
            e.printStackTrace();
            Toast.makeText(this, "Archivo no encontrado: " + "smile_detection_model.tflite", Toast.LENGTH_SHORT).show();
            throw e;
        }
    }
}