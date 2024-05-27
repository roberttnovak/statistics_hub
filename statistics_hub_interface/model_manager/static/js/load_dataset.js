$(document).ready(function() {
    var csrftoken = $('meta[name="csrf-token"]').attr('content');
    var selectedDataset = null;
    var selectedPath = null;

    $('.folder-name').on('click', function() {
        $(this).siblings('ul').toggle();
        $(this).children('.fas').toggleClass('fa-folder fa-folder-open');
    });
    
    $('.file').on('click', function(event) {
        event.stopPropagation();  // Prevenir la propagación para no desencadenar eventos de carpetas padre
        var fileInfo = gatherFileInfo($(this));
        
        // Eliminar la clase de todos los archivos para remover la selección anterior
        $('.file').removeClass('file-selected');
        // Añadir la clase al archivo actual para marcarlo como seleccionado
        $(this).addClass('file-selected');
    
        selectedDataset = fileInfo.dataset;  // Guardar el dataset seleccionado globalmente
        selectedPath = fileInfo.relativePath; // Guardar la ruta relativa seleccionada globalmente
        console.log("Selected Dataset:", selectedDataset); // Debugging para ver el dataset seleccionado
        console.log("Selected Path:", selectedPath); // Debugging para ver la ruta relativa seleccionada
    
        loadCsvPreview(fileInfo.dataset, fileInfo.relativePath);
    });

    function gatherFileInfo($fileElement) {
        var pathComponents = [];
        $fileElement.parentsUntil('.folder-view', 'li').each(function() {
            var directory = $(this).children('span.folder-name').text().trim();
            if (directory) {
                pathComponents.unshift(directory);
            }
        });
        var dataset = $fileElement.find('span').data('dataset');
        var relativePath = pathComponents.join('/');
        return { dataset: dataset, relativePath: relativePath };
    }

    function loadDataset(dataset) {
        var options = {};
        $('#load-dataset-options input, #load-dataset-options select').each(function() {
            var key = $(this).attr('id').replace('-input', '');
            var value = $(this).val();
            options[key] = value;
        });
    
        var queryParams = $.param(options);
    
        window.location.href = '/preprocess_dataset/' + encodeURIComponent(dataset) + '?' + queryParams;
    }

    $('#add-document-btn').click(function() {
        window.open('/upload_file/', '_blank');
    });

    $('#load-dataset-btn').click(function() {
        if (!selectedDataset ) {
            alert("Please select a dataset.");
            return;
        }
    
        var options = gatherOptions();
        var queryParams = $.param(options);
        var fullPath = selectedPath ? selectedPath + '/' + selectedDataset : selectedDataset;
        window.location.href = '/preprocess_dataset/' + encodeURIComponent(fullPath) + '?' + queryParams;
    });

    function gatherOptions() {
        var options = {};
        $('#load-dataset-options input, #load-dataset-options select').each(function() {
            var key = $(this).attr('id').replace('-input', '');
            var value = $(this).val();
            options[key] = value;
        });
        return options;
    }


    // Activar los tooltips de Bootstrap
    $('[data-toggle="tooltip"]').tooltip();

    // Botón para mostrar/ocultar las opciones de CSV
    // $('#toggle-options-btn').click(function() {
    //     $('#load-dataset-options').toggle();
    //     // Cambia el ícono del ojo de abierto a cerrado y viceversa
    //     $(this).find('i').toggleClass('far fa-eye far fa-eye-slash');
    //     // Actualiza el tooltip en función del estado del ícono
    //     var newTitle = $(this).find('i').hasClass('fa-eye') ? "Click to toggle visibility of import options" : "Click to hide import options";
    //     $(this).attr('data-original-title', newTitle).tooltip('show');
    // });


    // Handle folder creation
    $('#create-folder-form').on('submit', function(event) {
        event.preventDefault();
        var folderName = $('#folder-name-input').val();
        if (folderName) {
            $.ajax({
                url: '/create_folder/',
                type: 'POST',
                headers: {
                    'X-Requested-With': 'XMLHttpRequest',
                    'X-CSRFToken': csrftoken
                },
                data: { folder_name: folderName },
                success: function(response) {
                    // Handle success (e.g., reload file explorer or show a message)
                    alert('Folder created successfully!');
                },
                error: function(error) {
                    console.error("Error creating folder: ", error);
                }
            });
        } else {
            alert("Please enter a folder name.");
        }
    });

    // Handle file upload
    // $('#upload-form').on('submit', function(event) {
    //     event.preventDefault();
    //     var formData = new FormData(this);
    //     formData.append('action', 'preview_csv');
    //     $.ajax({
    //         url: '/load_dataset/',
    //         type: 'POST',
    //         headers: {
    //             'X-Requested-With': 'XMLHttpRequest',
    //             'X-CSRFToken': csrftoken
    //         },
    //         data: formData,
    //         processData: false,
    //         contentType: false,
    //         success: function(response) {
    //             alert('File uploaded successfully!');
    //         },
    //         error: function(error) {
    //             console.error("Error uploading file: ", error);
    //         }
    //     });
    // });

    // Ocultar elementos específicos al hacer clic en la pestaña "Upload File"
    $('#upload-tab').on('shown.bs.tab', function() {
        $('#load-dataset-btn').hide(); // Ocultar el botón "Load Dataset"
        $('#fileTypeTabs').hide(); // Ocultar las pestañas de csv, xlsx, etc.
        $('#csv-options').hide(); // Ocultar las opciones de CSV
        $('#excel-options').hide(); // Ocultar las opciones de Excel
        $('#sav-options').hide(); // Ocultar las opciones de SAV
        $('#json-options').hide(); // Ocultar las opciones de JSON
    });

    // Mostrar todo de nuevo al cambiar a otra pestaña
    $('a[data-toggle="tab"]').not('#upload-tab').on('shown.bs.tab', function() {
        $('#load-dataset-btn').show(); // Mostrar el botón "Load Dataset"
        $('#fileTypeTabs').show(); // Mostrar las pestañas de csv, xlsx, etc.
        $('#csv-options').show(); // Mostrar las opciones de CSV
        $('#excel-options').show(); // Mostrar las opciones de Excel
        $('#sav-options').show(); // Mostrar las opciones de SAV
        $('#json-options').show(); // Mostrar las opciones de JSON
    });


    $('#toggle-explanation-btn').click(function() {
        $('#explanation-section').toggle();
        var buttonText = $('#explanation-section').is(':visible') ? 'Hide Instructions' : 'Show Instructions';
        $(this).html('<i class="fas fa-info-circle"></i> ' + buttonText);
    });

    $('.delete-file').on('click', function() {
        var fileName = $(this).data('file');
        $.ajax({
            url: '/load_dataset/',
            method: 'POST',
            data: {
                'file_name': fileName,
                'action' : 'delete_file',
                'csrfmiddlewaretoken': '{{ csrf_token }}'
            },
            success: function(response) {
                location.reload();  // Recargar la página para actualizar la vista
            },
            error: function(response) {
                alert('Error deleting file');
            }
        });
    });

    $('#preview-csv-rows-btn').on('click', function() {
        if (!selectedDataset) {
            alert("Please select a dataset.");
            return;
        }
    
        var $previewSection = $('#csv-raw-preview');
        var $previewHeader = $('#csv-raw-preview-header');
    
        if ($previewSection.is(':visible')) {
            $previewSection.hide();
            $previewHeader.hide();
        } else {
            $.ajax({
                url: '/load_dataset/',
                type: 'POST',
                headers: {
                    'X-Requested-With': 'XMLHttpRequest',
                    'X-CSRFToken': csrftoken
                },
                data: {
                    action: 'preview_csv',
                    dataset: selectedDataset,
                    relativePath: selectedPath
                },
                success: function(response) {
                    $('#csv-raw-preview').html(response.raw_preview_html);
                    $previewHeader.show();
                    $previewSection.show();
                },
                error: function(error) {
                    console.error("Error previewing file: ", error);
                }
            });
        }
    });    
});