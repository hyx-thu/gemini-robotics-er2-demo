## PT Head 

This is meant to act as a tool for HRI R&D while not imposing on our limited robot resources and being extremely simple & cheap to reproduce. It has a mic, camera, and speaker with the ability to move in 2 axis. This is perfect for collecting audio-video data and testing and validating potential interactions. The platform makes it easy to acquire image and audio data and stream to whatever version of Gemini, Astra, or any VLA Model.

### Installation

To set up the environment and install the package, follow these steps:

1.  **Install system-level dependencies:**
    This script installs necessary system packages (such as `libportaudio2` for audio playback and development tools) required by the Python packages.
    ```bash
    sudo ./install_deps.sh
    ```

2.  **Source the activation script:**
    This script will create a local Python virtual environment (`.venv`) if it doesn’t exist and activate it for your current terminal session.
    ```bash
    source activate.sh
    ```

3.  **Install the Python package in editable mode:**
    ```bash
    pip install -e .
    ```
4. **Validate Install**
    > **⚠️ Warning:** This will cause the robot to move. After running this the robot should cycel through some of its gestures.
    ```bash
    python emotes.py
    ```

### Troubleshooting

- **`OSError: PortAudio library not found`**
  If you see this error when running scripts (like `examples/emotes.py`), it means the `libportaudio2` package was not installed on your system. Make sure you have run Step 1 (`sudo ./install_deps.sh`) successfully.

- **Permission denied on `/dev/ttyUSB0`**
  If the script fails to open the serial port to communicate with the hardware, you likely need to add your user to the `dialout` group:
  ```bash
  sudo usermod -a -G dialout $USER
  ```
  *(Note: You will need to log out and log back in for this change to take effect.)*

### Archived README

The previous version of the README with older instructions can be found in [README.old.md](README.old.md).

---

## Build Instructions Manual (BIM)

Ready to build the hardware? Because this process includes numerous step-by-step assembly photos, we've organized the hardware instructions into a dedicated manual.

👉 **[Click here to view the PT Head Assembly Manual](BIM.md)**

---

## Bill of Materials (BOM)

Below is the complete list of hardware required for this build. You can use the search bar to find specific components, sort by quantity, or filter as needed.

<!-- Interactive BOM Table HTML/JS below -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
<link rel="stylesheet" href="https://cdn.datatables.net/1.13.4/css/dataTables.bootstrap5.min.css">

<!-- FIX: Force the global background to respect VS Code's dark theme -->
<style>
  body {
    background-color: transparent !important;
    color: var(--vscode-editor-foreground) !important;
  }
</style>

<!-- FIX: Added data-bs-theme="dark" so the table styling matches the dark background -->
<div id="table-container" class="mt-4 mb-5" data-bs-theme="dark">
  <em>Loading BOM data...</em>
</div>

<!-- (Keep your scripts and JS below exactly as they were) -->
<script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
...
<script src="https://cdn.datatables.net/1.13.4/js/jquery.dataTables.min.js"></script>
<script src="https://cdn.datatables.net/1.13.4/js/dataTables.bootstrap5.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/PapaParse/5.3.2/papaparse.min.js"></script>

<script>
  $(document).ready(function() {
    // Ensure 'bom.csv' is uploaded to the root of your repository
    Papa.parse("bom.csv", {
      download: true,
      header: true,
      complete: function(results) {
        let data = results.data;
        let headers = results.meta.fields;
        
        let tableHtml = '<table id="bom-table" class="table table-striped table-bordered">';
        tableHtml += '<thead><tr>';
        headers.forEach(h => tableHtml += `<th>${h}</th>`);
        tableHtml += '</tr></thead><tbody>';
         
        data.forEach(row => {
          if (Object.values(row).some(x => x !== "")) { 
            tableHtml += '<tr>';
            headers.forEach(h => tableHtml += `<td>${row[h] || ''}</td>`);
            tableHtml += '</tr>';
          }
        });
        tableHtml += '</tbody></table>';
        
        $('#table-container').html(tableHtml);
        $('#bom-table').DataTable({ 
          "pageLength": 15, 
          "responsive": true,
          "order": [] // Maintains the original CSV order by default
        });
      }
    });
  });
</script>