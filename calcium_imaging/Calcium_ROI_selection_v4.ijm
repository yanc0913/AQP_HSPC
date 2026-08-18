// ==============================================================================
// Calcium ROI Selection Wizard for Dorsal Aorta Imaging (v4)
// ==============================================================================
//
// Purpose:
//   Semi-automated wizard for manual ROI selection on dual-channel (GCaMP + Lifeact)
//   calcium imaging time-series. Guides user through drawing:
//     - DA_band_dorsal: Dorsal dorsal aorta (thick segmented line)
//     - DA_band: Ventral dorsal aorta (thick segmented line)
//     - BG: Background region (rectangle)
//     - flat_01, flat_02, ...: Elongated endothelial cells (rectangles)
//     - round_01, round_02, ...: Round endothelial cells (rectangles)
//
// Input:
//   - Open hyperstack in Fiji/ImageJ (T×Z×C or Z×T×C format)
//   - At least 2 channels: GCaMP (calcium indicator) + Lifeact (actin marker)
//   - At least 2 timepoints
//
// Output:
//   1) *_ROIs.zip: Saved ROI set for ImageJ (all ROIs, not frame-specific)
//   2) *_roi_timeseries_allChannels.csv: Time-series measurements with columns:
//        image, phase, channel, roi_name, frame, mean, max, area,
//        bg_mean, mean_bgsub, max_bgsub
//
// Notes:
//   - Uses Lifeact channel for ROI drawing (better vessel visualisation)
//   - ROIs are global (not tied to specific frames) via Roi.setPosition(0)
//   - Background subtraction applied: mean_bgsub = mean - bg_mean
//   - (optional) Thick segmented lines converted to area ROIs for correct mean/area calculation
//
// See README.md for detailed workflow and downstream analysis with Python scripts.
//
// ==============================================================================


macro "DA_Calcium_ROI_Wizard_and_Export_v4" {

    // ===== Input validation =====
    if (nImages() == 0) exit("No image open.");

    outDir = getDirectory("Choose output folder");
    if (outDir == "") exit("No output folder chosen.");

    imgTitle = getTitle();
    Stack.getDimensions(w, h, nC, nZ, nT);

    if (nT < 2) exit("This image has < 2 time frames. Load a time series (hyperstack).");
    if (nC < 2) exit("This image has < 2 channels. Need at least GCaMP + Lifeact.");

    // Ask channel mapping
    Dialog.create("Channel mapping");
    Dialog.addMessage("Select channel indices (1-based) for your hyperstack:");
    Dialog.addNumber("GCaMP channel index (e.g., 1):", 1);
    Dialog.addNumber("Lifeact channel index (e.g., 2):", 2);
    Dialog.show();
    chG = Dialog.getNumber();
    chR = Dialog.getNumber();

    if (chG < 1 || chG > nC) exit("Invalid GCaMP channel index.");
    if (chR < 1 || chR > nC) exit("Invalid Lifeact channel index.");

    // File base name
    base = imgTitle;
    dot = lastIndexOf(base, ".");
    if (dot != -1) base = substring(base, 0, dot);

    roiZipPath = outDir + base + "_ROIs.zip";
    csvPath    = outDir + base + "_roi_timeseries_allChannels.csv";

    // Reset ROI Manager
    roiManager("Reset");

    // Switch to Lifeact channel for ROI drawing
    Stack.setChannel(chR);

    // --- Step 0: DA band dorsal (segmented line, thick) ---
    setTool("polyline"); // segmented line tool
    waitForUser(
        "Step 1/5: Draw DA_band_DORSAL as a THICK segmented line on Lifeact\n\n" +
        "1) Set Line Width (Edit > Options > Line Width...) e.g. 20 px.\n" +
        "2) Draw along DORSAL side of the DA.\n\n" +
        "Then click OK."
    );

    if (selectionType() == -1) exit("No selection found. Please draw DA_band and re-run.");

    // Convert thick line into AREA selection
    //run("Line to Area");

    // IMPORTANT: make ROI global (not tied to a specific frame)
    Roi.setPosition(0);

    roiManager("Add");
    roiManager("Select", roiManager("count")-1);
    roiManager("Rename", "DA_band_dorsal");


    // --- Step 1: DA band (segmented line, thick) ---
    setTool("polyline"); // segmented line tool
    waitForUser(
        "Step 2/5: Draw DA_band as a THICK segmented line on Lifeact\n\n" +
        "1) Set Line Width (Edit > Options > Line Width...) e.g. 20 px.\n" +
        "2) Draw along VENTRAL side of the DA where EHT happens.\n\n" +
        "Then click OK."
    );

    if (selectionType() == -1) exit("No selection found. Please draw DA_band and re-run.");

    // Convert thick line into AREA selection
    //run("Line to Area"); // uncomment if change line to area 

    // IMPORTANT: make ROI global (not tied to a specific frame)
    Roi.setPosition(0);

    roiManager("Add");
    roiManager("Select", roiManager("count")-1);
    roiManager("Rename", "DA_band");

    // --- Step 2: Background ROI ---
    setTool("rectangle");
    waitForUser(
        "Step 3/5: Draw BG ROI (background)\n\n" +
        "Rectangle tool is set.\n" +
        "Draw a background ROI near the DA, avoiding cells.\n\n" +
        "Then click OK."
    );

    if (selectionType() == -1) exit("No selection found. Please draw BG ROI and re-run.");

    Roi.setPosition(0);

    roiManager("Add");
    roiManager("Select", roiManager("count")-1);
    roiManager("Rename", "BG");

    // --- Step 3: Cell ROIs loop ---
    flatCount = 0;
    roundCount = 0;

    while (true) {

        Dialog.create("Step 4/5: Add cell ROIs");
        Dialog.addMessage(
            "Draw cell ROIs on Lifeact (current channel).\n\n" +
            "Suggestion:\n" +
            "- Use Rectangle ROIs with a small margin.\n" +
            "- Avoid overlapping neighboring cells as much as possible."
        );
        Dialog.addChoice("Next ROI type:", newArray("flat cell", "round/budding cell", "Done"), "flat cell");
        Dialog.show();
        choice = Dialog.getChoice();

        if (choice == "Done") break;

        setTool("rectangle");

        if (choice == "flat cell") {
            flatCount++;
            roiName = "flat_" + d2(flatCount);
            msg = "Draw rectangle ROI for " + roiName + "\n\n" +
                  "Draw around ONE flat DA endothelial cell.\n" +
                  "Then click OK.";
        } else {
            roundCount++;
            roiName = "round_" + d2(roundCount);
            msg = "Draw rectangle ROI for " + roiName + "\n\n" +
                  "Draw around ONE round/budding cell.\n" +
                  "Then click OK.";
        }

        waitForUser(msg);

        if (selectionType() == -1) {
            if (choice == "flat cell") flatCount--;
            else roundCount--;
            exit("No selection found. Please draw the ROI when prompted.");
        }

        Roi.setPosition(0);

        roiManager("Add");
        roiManager("Select", roiManager("count")-1);
        roiManager("Rename", roiName);
    }

    // Save ROIs
    roiManager("Save", roiZipPath);

    // check ROI
    nR = roiManager("count");
    if (nR < 2) exit("Not enough ROIs. Need at least DA_band and BG (and ideally cells).");

    // Phase label
    Dialog.create("Phase label");
    Dialog.addChoice("Step 5/5: Which phase is this movie?", newArray("pre", "post"), "pre");
    Dialog.show();
    phase = Dialog.getChoice();

    // --- Step 4: Export time-series for BOTH channels (GCaMP + Lifeact) ---

    // Find BG ROI index
    bgIndex = -1;
    nR = roiManager("count");
    for (i = 0; i < nR; i++) {
        roiManager("Select", i);
        nm = Roi.getName();
        if (nm == "BG") bgIndex = i;
    }
    if (bgIndex < 0) exit("ROI named 'BG' not found (unexpected).");

    // If your movie is subMAX, nZ should be 1.
    zPlane = 1;

    // Prepare CSV
    if (File.exists(csvPath)) File.delete(csvPath);
    File.append("image,phase,channel,roi_name,frame,mean,max,area,bg_mean,mean_bgsub,max_bgsub\r\n", csvPath);

    run("Set Measurements...", "area mean min max redirect=None decimal=6");
    run("Clear Results");

    // loop two channels: GCaMP then Lifeact
    for (cc = 0; cc < 2; cc++) {

        if (cc == 0) { c = chG; chLabel = "GCaMP"; }
        else         { c = chR; chLabel = "Lifeact"; }

        for (t = 1; t <= nT; t++) {

            // Set stack position first
            Stack.setPosition(c, zPlane, t);

            // measure BG once for this frame
            roiManager("Select", bgIndex);
            // Re-assert in case ROI selection tries to move the stack
            Stack.setPosition(c, zPlane, t);

            run("Measure");
            bgMean = getResult("Mean", nResults-1);

            // loop ROIs
            for (r = 0; r < nR; r++) {

                roiManager("Select", r);
                // Re-assert again to guarantee correct (c,z,t)
                Stack.setPosition(c, zPlane, t);

                roiName = Roi.getName();

                run("Measure");
                meanVal = getResult("Mean", nResults-1);
                maxVal  = getResult("Max",  nResults-1);
                areaVal = getResult("Area", nResults-1);

                meanBgSub = meanVal - bgMean;
                maxBgSub  = maxVal  - bgMean;

                line = "\"" + imgTitle + "\"," + phase + "," + chLabel + "," + "\"" + roiName + "\"," +
                       t + "," + meanVal + "," + maxVal + "," + areaVal + "," + bgMean + "," +
                       meanBgSub + "," + maxBgSub;

                File.append(line + "\r\n", csvPath);
            }
        }
    }

    print("Saved ROIs: " + roiZipPath);
    print("Saved CSV:  " + csvPath);

    showMessage("Done",
        "ROIs saved:\n" + roiZipPath + "\n\n" +
        "Time-series CSV saved:\n" + csvPath + "\n\n" +
        "Now the CSV should show changing mean/max across frames.\n" +
        "Next: Python F/F0 + peak detection + summary plots.");
}


// helper: format 2-digit index
function d2(n) {
    if (n < 10) return "0" + n;
    return "" + n;
}
