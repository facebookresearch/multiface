# Data Structure
Here, we show an example of dataset structure of entity ```6795937```. Notice that each entity can have slightly different content (eg. different number of cameras, different number of frames under each expression), but the general structure should be preserved in the same way.

Under its root directory ```m−−20180227−−0000−−6795937−−GHS```, you should find:
```
* frame_list.txt
* KRT
* tex_mean.png
* tex_var.txt
* vert_mean.bin
* vert_var.txt

* images
  - E001_Neutral_Eyes_Open
     * 400002
       * 000102.png 
       * 000105.png 
       * (xxxxxx).png 
       * 000177.png
     * 400004
     * (camera_index)
     * 400070
  - E002_Swallow
  - (expressions)
  - SEN_youre_boiling_milk_aint_you
  
* unwrapped_uv_1024
  - E001_Neutral_Eyes_Open
     * 400002
       * 000102.png 
       * 000105.png 
       * (xxxxxx).png 
       * 000177.png
     * 400004
     * (camera_index)
     * 400070
  - E002_Swallow
  - (expressions)
  - SEN_youre_boiling_milk_aint_you
  
* tracked_mesh
  - E001_Neutral_Eyes_Open
     * 000102_transform.txt
     * 000102.bin
     * 000102.obj
     * 000105_transform.txt
     * 000105.bin
     * 000105.obj
     * (xxxxxx)_transform.txt
     * (xxxxxx).bin
     * (xxxxxx).obj
     * 000177_transform.txt
     * 000177.bin
     * 000177.obj
  - E002_Swallow
  - (expressions)
  - SEN_youre_boiling_milk_aint_you

* audio
  - SEN_a_good_morrow_to_you_my_boy.wav
  - SEN_a_voice_spoke_near-at-hand.wav
  - (segment expressions)
  - SEN_youre_boiling_milk_aint_you.wav
```
The mapping between file/<ins>folder</ins> names and dataset asset is as the following:

| File/<ins>Folder</ins> Name       | Dataset Asset        | 
| ------------- |:-------------:|
| <ins>images</ins>   | Raw Images |
| <ins>unwrapped_uv_1024</ins> |Unwrapped Textures   |   
| <ins>tracked_mesh</ins>   | Tracked Meshes    | 
| (xxxxxx)_transform.txt|Headpose     |   
| <ins>audio</ins>|Audio     |    
|KRT |Metdata - camera calibration     |  
| frame_list.txt|Metadata - frame list    |  
|tex_mean.png|Metadata - texture mean   |  
| tex_var.txt|Metadata - texture variance    |  
| vert_mean.bin|Metadata - vertex mean    |  
| vert_var.txt|Metadata - vertex variance    |  
