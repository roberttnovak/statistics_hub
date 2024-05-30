$(document).ready(function() {
    var csrftoken = $('meta[name="csrf-token"]').attr('content');
    var selectedDataset = null;
    var selectedPath = null;

    // Function to gather file info (dataset and relative path)
    function gatherFileInfo($element) {
        var pathComponents = [];
        $element.parentsUntil('.folder-view', 'li').each(function() {
            var directory = $(this).children('span.folder-name').first().text().trim();
            if (directory) {
                pathComponents.unshift(directory);
            }
        });
        var dataset = $element.data('dataset') || $element.text().trim();
        var relativePath = pathComponents.join('/');
        return { dataset: dataset, relativePath: relativePath };
    }

    // Function to gather options from form inputs
    function gatherOptions() {
        var options = {};
        $('#load-dataset-options input, #load-dataset-options select').each(function() {
            var key = $(this).attr('id').replace('-input', '');
            var value = $(this).val();
            options[key] = value;
        });
        return options;
    }

    // Function to load dataset
    function loadDataset(dataset) {
        var options = gatherOptions();
        var queryParams = $.param(options);
        window.location.href = '/preprocess_dataset/' + encodeURIComponent(dataset) + '?' + queryParams;
    }

    // Handle folder click event to toggle visibility of child elements
    $('.folder-name').on('click', function() {
        $(this).siblings('ul').toggle();
        $(this).children('.fas').toggleClass('fa-folder fa-folder-open');
        $(this).siblings('.folder-actions').toggleClass('d-none');
    
        // Update the selectedPath to the path of the clicked folder
        var fileInfo = gatherFileInfo($(this).closest('li'));
        selectedPath = fileInfo.relativePath;
        console.log("Updated Path:", selectedPath); // Debugging to see the updated path
    });

    // Handle file click event to select the file and load its preview
    $('.file').on('click', function(event) {
        event.stopPropagation();  // Prevent propagation to avoid triggering parent folder events
        var fileInfo = gatherFileInfo($(this));
        
        // Remove selection from all files and add to the clicked one
        $('.file').removeClass('file-selected');
        $(this).addClass('file-selected');
    
        selectedDataset = fileInfo.dataset;  // Save the selected dataset globally
        selectedPath = fileInfo.relativePath; // Save the selected relative path globally
        console.log("Selected Dataset:", selectedDataset); // Debugging to see the selected dataset
        console.log("Selected Path:", selectedPath); // Debugging to see the selected relative path
    
    });

    // Handle dataset load button click
    $('#load-dataset-btn').click(function() {
        if (!selectedDataset) {
            alert("Please select a dataset.");
            return;
        }
    
        var options = gatherOptions();
        var queryParams = $.param(options);
        var fullPath = selectedPath ? selectedPath + '/' + selectedDataset : selectedDataset;
        window.location.href = '/preprocess_dataset/' + encodeURIComponent(fullPath) + '?' + queryParams;
    });

    // Handle document upload button click
    $('#add-document-btn').click(function() {
        window.open('/upload_file/', '_blank');
    });

    // Handle folder creation form submission
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

    // Handle file deletion
    $('.delete-file').click(function() {
        // It is necessary to select the file before deleting it
        var $fileElement = $(this).closest('.file');
        $fileElement.click();  // Trigger the file click event to select the file
        
        var $selectedFile = $('.file-selected');
    
        var fileInfo = gatherFileInfo($selectedFile);
        selectedDataset = fileInfo.dataset;  
        selectedPath = fileInfo.relativePath; 
    
        if (confirm("Are you sure you want to delete this file?")) {
            $.ajax({
                type: 'POST',
                url: '/load_dataset/',
                headers: {
                    'X-Requested-With': 'XMLHttpRequest',
                    'X-CSRFToken': csrftoken
                },
                data: {
                    'action': 'delete_file',
                    'file_name': selectedDataset,
                    'relativePath': selectedPath
                },
                success: function(response) {
                    alert("File deleted successfully");
                    location.reload();
                },
                error: function(xhr, status, error) {
                    alert("An error occurred: " + xhr.responseText);
                }
            });
        }
    });

    // Handle CSV preview button click
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

    // Handle delete folder button click
    $(document).on('click', '.delete-folder', function() {
        var $folderElement = $(this).closest('li').children('.folder-name');
        var fileInfo = gatherFileInfo($folderElement);
        
        if (confirm("Are you sure you want to delete this folder?")) {
            $.ajax({
                type: 'POST',
                url: '/load_dataset/',
                headers: {
                    'X-Requested-With': 'XMLHttpRequest',
                    'X-CSRFToken': csrftoken
                },
                data: {
                    'action': 'delete_folder',
                    'folder_name': fileInfo.dataset,
                    'relativePath': fileInfo.relativePath
                },
                success: function(response) {
                    alert("Folder deleted successfully");
                    location.reload();
                },
                error: function(xhr, status, error) {
                    alert("An error occurred: " + xhr.responseText);
                }
            });
        }
    });


    // Handle folder creation form submission inside file explorer
    document.querySelectorAll(".add-folder").forEach(button => {
        button.addEventListener("click", function() {
            const folderActions = this.closest("li").querySelector(".folder-actions");
            const createFolderInput = this.closest("li").querySelector(".create-folder-input");
            
            if (createFolderInput.classList.contains("d-none")) {
                createFolderInput.classList.remove("d-none");
                folderActions.classList.add("d-none");
            } else {
                createFolderInput.classList.add("d-none");
                folderActions.classList.remove("d-none");
            }
        });
    });

    document.querySelectorAll(".submit-folder").forEach(button => {
        button.addEventListener("click", function() {
            const input = this.closest("li").querySelector(".folder-name-input");
            const folderName = input.value;
            const path = this.getAttribute("data-path");
    
            if (folderName.trim()) {
                $.ajax({
                    url: '/load_dataset/',
                    type: 'POST',
                    headers: {
                        'X-Requested-With': 'XMLHttpRequest',
                        'X-CSRFToken': csrftoken
                    },
                    data: {
                        action: 'create_folder',
                        folder_name: folderName,
                        relativePath: selectedPath ? selectedPath + '/' + path : path
                    },
                    success: function(response) {
                        alert('Folder created successfully!');
                        location.reload();  // Reload the page to reflect changes
                    },
                    error: function(error) {
                        console.error("Error creating folder: ", error);
                        alert('An error occurred while creating the folder.');
                    }
                });
            }
        });
    });

    // Handle cancel button click to hide input and show folder/file actions
    document.querySelectorAll(".cancel-folder").forEach(button => {
        button.addEventListener("click", function() {
            const folderActions = this.closest("li").querySelector(".folder-actions");
            const createFolderInput = this.closest("li").querySelector(".create-folder-input");

            createFolderInput.classList.add("d-none");
            folderActions.classList.remove("d-none");
        });
    });

    $('.file-explorer > .add-folder').on('click', function() {
        const createFolderInput = $(this).siblings('.create-folder-input');
        const addFileButton = $(this).siblings('.add-file');
        const addFolderButton = $(this);
    
        if (createFolderInput.hasClass('d-none')) {
            createFolderInput.removeClass('d-none');
            addFileButton.addClass('d-none');
            addFolderButton.addClass('d-none');
        } else {
            createFolderInput.addClass('d-none');
            addFileButton.removeClass('d-none');
            addFolderButton.removeClass('d-none');
        }
    });

    // Manejador para cancelar la creación en el root
    $('.file-explorer > .create-folder-input .cancel-folder').on('click', function() {
        const createFolderInput = $(this).closest('.create-folder-input');
        const addFileButton = $(this).closest('.file-explorer').find('.add-file');
        const addFolderButton = $(this).closest('.file-explorer').find('.add-folder');
    
        createFolderInput.addClass('d-none');
        addFileButton.removeClass('d-none');
        addFolderButton.removeClass('d-none');
    });

    // Manejador para enviar la creación de carpeta en el root
    $('.file-explorer > .create-folder-input .submit-folder').on('click', function() {
        const input = $(this).siblings('.folder-name-input');
        const folderName = input.val();
        const path = $(this).data('path');

        if (folderName.trim()) {
            $.ajax({
                url: '/load_dataset/',
                type: 'POST',
                headers: {
                    'X-Requested-With': 'XMLHttpRequest',
                    'X-CSRFToken': csrftoken
                },
                data: {
                    action: 'create_folder',
                    folder_name: folderName,
                    relativePath: path
                },
                success: function(response) {
                    alert('Folder created successfully!');
                    location.reload();
                },
                error: function(error) {
                    console.error("Error creating folder: ", error);
                    alert('An error occurred while creating the folder.');
                }
            });
        }
    });

    // Handle opening the file
    $('.file-explorer').on('click', '.add-file', function() {
        var inputFile = $(this).siblings('.upload-files-input');
        var dataPath = $(this).data('path');
        inputFile.attr('data-path', dataPath);
        inputFile.click();
    });    

    // Handle file upload
    $('.file-explorer').on('change', '.upload-files-input', function() {
        var files = this.files;
        var formData = new FormData();
        var dataPath = $(this).attr('data-path');  // Usar attr en lugar de data
        
        // Concatenar selectedPath y dataPath, asegurando que no haya duplicados de carpeta
        var fullPath = selectedPath ? selectedPath + '/' + dataPath : dataPath;
    
        // Limpiar el fullPath para evitar duplicados innecesarios
        fullPath = fullPath.replace(/\/{2,}/g, '/');  // Reemplaza múltiples barras con una sola
    
        // Remover barra inicial si existe
        if (fullPath.startsWith('/')) {
            fullPath = fullPath.substring(1);
        }
    
        console.log("dataPath:", dataPath); // Debugging to see the data path
        console.log("selectedPath:", selectedPath); // Debugging to see the selected path
        console.log("fullPath:", fullPath); // Debugging to see the full path
    
        formData.append('relativePath', fullPath);
        formData.append('action', 'upload_files');
    
        for (var i = 0; i < files.length; i++) {
            formData.append('files', files[i]);
        }
    
        $.ajax({
            url: '',
            type: 'POST',
            headers: {
                'X-Requested-With': 'XMLHttpRequest',
                'X-CSRFToken': csrftoken
            },
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                alert('Files uploaded successfully!');
                location.reload();
            },
            error: function(error) {
                console.error("Error uploading files: ", error);
                alert('An error occurred while uploading the files.');
            }
        });
    });

    $('#fileManagementTabs a[data-toggle="tab"]').on('shown.bs.tab', function (e) {
        var target = $(e.target).attr("href"); // Obtener el ID de la pestaña activa
        
        if (target === "#file-explorer-tab") {
            $('#file-explorer-options').show();
            $('#mysql-file-tab').hide();
        } else {
            $('#file-explorer-options').hide();
            $('#mysql-file-tab').show();
        }
    });

    // Handle the toggle button for the explanation section
    $('#toggle-explanation-btn').click(function() {
        $('#explanation-section').toggle();
        var buttonText = $('#explanation-section').is(':visible') ? 'Hide Instructions' : 'Show Instructions';
        $(this).html('<i class="fas fa-info-circle"></i> ' + buttonText);
    });

    // Activate Bootstrap tooltips
    $('[data-toggle="tooltip"]').tooltip();

    $('.mysql-section-title').on('click', function() {
        $(this).next('.mysql-section-content').toggleClass('active');
        $(this).find('.toggle-icon i').toggleClass('fa-chevron-down fa-chevron-up');
    });
    
});
