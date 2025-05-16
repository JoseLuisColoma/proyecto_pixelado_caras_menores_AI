function uploadImage() {
    const input = document.getElementById('imageInput');
    const status = document.getElementById('status');
    const outputImage = document.getElementById('outputImage');

    if (!input.files.length) {
        status.textContent = "Por favor, selecciona una imagen.";
        status.style.color = "red";
        return;
    }

    const formData = new FormData();
    formData.append("image", input.files[0]);

    status.textContent = "Procesando la imagen...";
    status.style.color = "steelblue";
    outputImage.style.display = "none";

    fetch("/process", {
        method: "POST",
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error("Error al procesar la imagen");
        }
        return response.blob();
    })
    .then(blob => {
        const imageUrl = URL.createObjectURL(blob);
        outputImage.src = imageUrl;
        outputImage.style.display = "block";
        status.textContent = "¡Imagen correctamente procesada!";
        status.style.color = "green";
    })
    .catch(error => {
        console.error("Error:", error);
        status.textContent = "Error: " + error.message;
        status.style.color = "red";
    });
}
