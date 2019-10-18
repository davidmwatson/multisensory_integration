# Parse commandline arguments
Param(
    [string]$videodir,
    [string]$audiodir,
    [string]$outdir,
    [string]$videoext=".mp4",
    [string]$audioext=".wav",
    [switch]$help
)

if ( $help ) {
    Write-Host("Script uses ffmpeg to mux together a set of video files with
a corresponding set of audio files.  Note that filenames (barring extensions)
must be the same for each video / audio file pair, and that every video
file must have a corresponding audio file.

Arguments
---------
-videodir : Directory containing video files
-audiodir : Directory containing audio files
-outdir : Desired output directory, will be created if necessary
-videoext (optional) : Extension of video files, defaults to .mp4
-audioext (optional) : Extension of audio files, defaults to .wav
-help : Print help message and exit")
    exit
}

# Create outdir if necessary
if ( -not $( test-path $outdir ) ) {
    new-item $outdir -type directory | out-null
}

# Search for files, error if none found
$vidfiles = $( get-childitem $videodir *$videoext | sort-object name )
$audfiles = $( get-childitem $audiodir *$audioext | sort-object name )
if ( -not $( $vidfiles.length -gt 0 ) ) {
    throw "Failed to find any video files"
}
if ( -not $( $audfiles.length -gt 0 ) ) {
    throw "Failed to find any audio files"
}
if ( $vidfiles.count -ne $audfiles.count ) {
    throw "Number of video files does not match number of audio files"
}

# Loop
for ( $i=0; $i -lt $vidfiles.count; $i++ ) {
    echo $i
    $vidfile = $vidfiles[$i]
    $audfile = $audfiles[$i]
    if ( $vidfile.basename -ne $audfile.basename ) {
        throw "Video file ($($vidfile.name)) does not match audio file ($($audfile.name))"
    }

    # Mux
    write-host "`n`n### Muxing $vidfile and $audfile ###"
    $vid_codec = "copy"
    if ($videoext -eq ".mp4") {$aud_codec = "aac"} else {$aud_codec = "copy"}
    ffmpeg.exe -y -i $vidfile.fullname -i $audfile.fullname -c:v $vid_codec `
        -c:a $aud_codec $( join-path $outdir $vidfile.name )
}
