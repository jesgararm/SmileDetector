package us.aplicaciones.smiledetector;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.provider.MediaStore;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.DataType;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Objects;

public class MainActivity extends AppCompatActivity {

    private static final int REQUEST_IMAGE_CAPTURE = 1;
    private static final int REQUEST_SELECT_IMAGE = 2;
    private static final int PERMISSION_REQUEST_CODE = 100;

    private ImageView imageView;
    private TextView textResult;
    private Interpreter tfliteInterpreter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = findViewById(R.id.image_view);
        textResult = findViewById(R.id.text_result);

        Button btnTakePhoto = findViewById(R.id.btn_take_photo);
        Button btnSelectGallery = findViewById(R.id.btn_select_gallery);

        btnTakePhoto.setOnClickListener(v -> capturePhoto());
        btnSelectGallery.setOnClickListener(v -> selectFromGallery());

        checkPermissions();

        try {
            tfliteInterpreter = new Interpreter(loadModelFile());
            Toast.makeText(this, "Modelo cargado correctamente", Toast.LENGTH_SHORT).show();
        } catch (Exception e) {
            e.printStackTrace();
            Toast.makeText(this, "Error al cargar el modelo", Toast.LENGTH_LONG).show();
        }
    }

    @SuppressLint("NewApi")
    private void checkPermissions() {
        String[] permissions = {
                Manifest.permission.CAMERA,
                Manifest.permission.READ_EXTERNAL_STORAGE
        };

        for (String permission : permissions) {
            if (checkSelfPermission(permission) != PackageManager.PERMISSION_GRANTED) {
                requestPermissions(permissions, PERMISSION_REQUEST_CODE);
                return;
            }
        }
    }

    @SuppressLint("QueryPermissionsNeeded")
    private void capturePhoto() {
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
            startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE);
        } else {
            Toast.makeText(this, "No se pudo abrir la cámara", Toast.LENGTH_SHORT).show();
        }
    }

    private void selectFromGallery() {
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(intent, REQUEST_SELECT_IMAGE);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode == RESULT_OK && data != null) {
            Bitmap bitmap = null;
            try {
                if (requestCode == REQUEST_IMAGE_CAPTURE) {
                    bitmap = (Bitmap) Objects.requireNonNull(data.getExtras()).get("data");
                } else if (requestCode == REQUEST_SELECT_IMAGE) {
                    bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), data.getData());
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

    private void detectSmile(Bitmap bitmap) {
        try {
            TensorImage tensorImage = preprocessImage(bitmap);

            if (tensorImage == null) {
                Toast.makeText(this, "Error al preprocesar la imagen", Toast.LENGTH_SHORT).show();
                return;
            }

            float[][] output = new float[1][1];
            tfliteInterpreter.run(tensorImage.getBuffer(), output);

            String result = output[0][0] > 0.5 ? "Sonrisa detectada!" : "No hay sonrisa.";
            textResult.setText(result);
        } catch (Exception e) {
            e.printStackTrace();
            Toast.makeText(this, "Error al realizar la predicción", Toast.LENGTH_SHORT).show();
        }
    }

    private TensorImage preprocessImage(Bitmap bitmap) {
        try {
            TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
            tensorImage.load(bitmap);

            ImageProcessor processor = new ImageProcessor.Builder()
                    .add(new ResizeOp(128, 128, ResizeOp.ResizeMethod.BILINEAR))
                    .build();

            return processor.process(tensorImage);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private MappedByteBuffer loadModelFile() throws IOException {
        try {
            FileInputStream inputStream = new FileInputStream(getAssets().openFd("smile_detection_model.tflite").getFileDescriptor());
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = inputStream.getChannel().position();
            long declaredLength = fileChannel.size();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        } catch (IOException e) {
            e.printStackTrace();
            Toast.makeText(this, "Archivo no encontrado: " + "smile_detection_model.tflite", Toast.LENGTH_SHORT).show();
            throw e;
        }
    }
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (requestCode == PERMISSION_REQUEST_CODE) {
            for (int result : grantResults) {
                if (result != PackageManager.PERMISSION_GRANTED) {
                    Toast.makeText(this, "Permisos no concedidos. La aplicación podría no funcionar correctamente.", Toast.LENGTH_LONG).show();
                    return;
                }
            }
        }
    }
}