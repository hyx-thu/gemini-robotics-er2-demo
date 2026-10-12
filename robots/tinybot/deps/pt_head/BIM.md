# PT Head - Build Instructions Manual (BIM)

Welcome to the assembly manual for the PT Head! This guide will walk you through preparing your 3D printed parts, configuring the Dynamixel servos, wiring the audio harness, and building the complete head assembly.

**Estimated Build Time:** 2-3 hours  
**Difficulty:** Intermediate  

---

## 🛠️ Tools Required
Before you begin, make sure you have the following tools on hand:
* Heat-set insert tool or soldering iron with heat-set tips
* Soldering iron & solder (for cable harness and amp assembly)
* Hex key set / Screwdrivers (specifically M2 and M3 sizes)
* Wire strippers / snips
* Super glue (for faux antennas)
* Marker / Fine-point Sharpie

---

## Phase 1: Preparation & Threaded Inserts

In this phase, you will print all necessary parts and install brass heat-set inserts to prepare the components for assembly.

> **💡 Pro Tip:** Check the BOM (Bill of Materials) file for the complete list of 3D-printed parts, required quantities, and recommended print settings.

### Step 1: Heat-Set Insert Installation
Using a heat-set insert tool or soldering iron, carefully press the brass threaded inserts into their designated locations until flush with the plastic surface:

* **Helmet:** 6 inserts
* **mic-array-case-bottom:** 4 inserts
* **mic-array-case-top:** 6 inserts
* **neck-p1:** 4 inserts
* **neck-base:** 4 inserts

---

## Phase 2: Dynamixel Servo Configuration & Sub-Assembly

Here you will configure the servo IDs and baud rates using the Dynamixel Wizard before mounting the initial neck components.

### Step 1: Connect and Configure Servos
1. Assemble the U2D2 board, then download and install **Dynamixel Wizard**.
2. Connect your motors to the U2D2 board. 

> **⚠️ Warning:** These servos have different torque values! Double-check their IDs before proceeding to prevent structural issues during operation.

3. Update the parameters for both servos to **Baud Rate: 115200** and zero them to position **180**.
   * **ID 1 (XL330-M288-T):** Neck Yaw (Pan)
   * **ID 2 (XC330-M288-T):** Neck Pitch (Tilt)

> **💡 Pro Tip:** Mark the 180 position on both the servo body and horn, and write the ID number directly on the servo body to avoid mix-ups during physical assembly.

### Step 2: Assemble Neck Servos
1. Attach the additional idler horn to the non-driven side of the **XC330 (ID 2)** servo.
2. Mount **neck-p2** and **neck-p3** to either side of the XC330 servo and secure using 4 x M2x6mm screws on each side.
3. Mount the **XL330 (ID 1)** into **neck-p1** and fasten it from underneath using 4 x M2x10mm screws.

---

## Phase 3: Electronics & Audio Harness Assembly

In this phase, you will solder the custom speaker harness and prepare the audio amplifier board.

### Step 1: Assemble Speaker Cable Harness
The audio amplifier requires two ribbon cables for communication (LRC, BCLK, DIN) and power (VIN, GND). Cable strand #1 is identified by the black wire.

1. **Ribbon Cable A (Data):** Isolate strands 1, 10, and 12. Cut away all unused strands.
2. **Ribbon Cable B (Power):** Isolate strands 1 and 2. Cut away all unused strands.
3. Solder the connections to the amplifier board as follows:
   * **VIN:** Ribbon Cable B — Wire #2
   * **GND:** Ribbon Cable B — Wire #1
   * **DIN:** Ribbon Cable A — Wire #1
   * **BCLK:** Ribbon Cable A — Wire #10
   * **LRC:** Ribbon Cable A — Wire #12

### Step 2: Assemble the Amp Board
1. Solder the pin headers into the amplifier board.
2. Solder the screw terminals onto the board for the speaker connections.

---

## Phase 4: Base & Neck Assembly

### Step 1: Base Mounts
1. Mount the **U2D2 board** to the base plate using 4 x M3x10mm screws.
2. Mount the **XC330 (ID 2)** servo into the base plate using 4 x M2x10mm screws.
3. Connect the 3-pin cable from the U2D2 board to the **XC330 (ID 2)** servo.

### Step 2: Neck Integration
1. Attach **neck-p1** to **neck-p2** and **neck-p3** using 4 x M3x6mm screws.
2. Attach the servo assembly to the **mic-array-case-bottom** (ensuring the USB port faces toward the top of the servo) using 4 x M2x10mm screws.

---

## Phase 5: Head Assembly & Camera Integration

### Step 1: Assemble the Microphone Array Case
1. Place the microphone array board into **mic-array-case-bottom**, ensuring the USB port is properly aligned.
2. Fasten the 2 x M3x6mm screws closest to the USB port into the bottom case (this forms the back of the head).
3. Take **mic-array-top-case** and pass the 2 ribbon cable ends from your previously prepared speaker harness through the slot.
4. Plug both ribbon cable ends into the appropriate headers on the mic array board.
5. Place **mic-array-top-case** onto **mic-array-case-bottom**.
6. Secure with 2 x M3x18mm screws to fully seal the mic array case.

### Step 2: Mount the Intel RealSense Camera
1. Mount the **realsense-mount** to the top of **mic-array-top-case** using 2 x M3x6mm screws.
2. Mount the **Intel RealSense D415** to the bracket using 2 x M3x6mm screws.
3. Connect the 90-degree USB-C cable and screw it into the D415 body to secure it.

### Step 3: Install Speaker & Amplifier
1. Mount the speaker into the **helmet** using 4 x M3x6mm screws, ensuring the speaker cables (+/-) point toward the amp mounting point.
2. Insert the lead wires from the speaker into the screw terminals on the amplifier board and tighten them down.
3. Secure the amplifier board to its in-helmet mounting point using 2 x M3x6mm screws.

### Step 4: Final Enclosure & Faux Antennas
1. Connect the speaker harness to the amplifier board.
2. Route the RealSense USB-C camera cable out through the slot in the back of the helmet.
3. Place the helmet onto the mic array case assembly, aligning it with the 4 mounting holes.
4. Secure the helmet using 4 x M3x6mm screws.
5. *(Optional)* Apply a drop of super glue to the decorative "faux antennas" and insert them into the helmet slots.

---

## 🎉 Completion! 
Congrats! Your PT Head unit is now fully assembled and wired.

Head back to the [Main Readme](index.md) to follow the software setup and launch instructions!