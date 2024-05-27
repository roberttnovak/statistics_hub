$(document).ready(function() {
    var csrftoken = $('meta[name="csrf-token"]').attr('content');
    var selectedDataset = null;
    var selectedPath = null;

    // Function to gather file info (dataset and relative path)
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

    // Handle tab visibility changes for upload section
    $('#upload-tab').on('shown.bs.tab', function() {
        $('#load-dataset-btn').hide(); // Hide the "Load Dataset" button
        $('#fileTypeTabs').hide(); // Hide file type tabs (csv, xlsx, etc.)
        $('#csv-options').hide(); // Hide CSV options
        $('#excel-options').hide(); // Hide Excel options
        $('#sav-options').hide(); // Hide SAV options
        $('#json-options').hide(); // Hide JSON options
    });

    // Show all options when switching to other tabs
    $('a[data-toggle="tab"]').not('#upload-tab').on('shown.bs.tab', function() {
        $('#load-dataset-btn').show(); // Show the "Load Dataset" button
        $('#fileTypeTabs').show(); // Show file type tabs (csv, xlsx, etc.)
        $('#csv-options').show(); // Show CSV options
        $('#excel-options').show(); // Show Excel options
        $('#sav-options').show(); // Show SAV options
        $('#json-options').show(); // Show JSON options
    });

    // Handle the toggle button for the explanation section
    $('#toggle-explanation-btn').click(function() {
        $('#explanation-section').toggle();
        var buttonText = $('#explanation-section').is(':visible') ? 'Hide Instructions' : 'Show Instructions';
        $(this).html('<i class="fas fa-info-circle"></i> ' + buttonText);
    });

    // Activate Bootstrap tooltips
    $('[data-toggle="tooltip"]').tooltip();
});
