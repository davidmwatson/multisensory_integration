# Parse commandline arguments
Param(
    [string]$videodir,
    [string]$videofile,
    [string]$outbase,
    [string]$videoext=".mp4",
    [switch]$help
)

if ( $help ) {
    Write-Host("Uses ffmpeg to split a set of video files into
their constituent frames.  Each set of frames will be saved in a directory
named after the filename of the corresponding video.  These directories in
turn will be created in the `$outbase directory if one is specified,
or in the `$videodir directory otherwise.

Arguments
---------
-videodir (optional): Directory containing video files, script will batch
                      process all files in this directory. Cannot be specified
                      along with -videofile flag.
-videofile (optional): Path to single video file to process.  Cannot be
                       specified along with -videodir flag.
-outbase (optional) : Desired output directory, defaults to `$videodir or
                      dirname of `$videofile.
-videoext (optional) : Extension of video files, defaults to .mp4.  Only
                       applicable if specifying a video directory.
-help : Print help message and exit")
    exit
}

$haveVideoDir = -not [string]::IsNullOrWhiteSpace($videodir)
$haveVideoFile = -not [string]::IsNullOrWhiteSpace($videofile)
$haveOutbase = -not [string]::IsNullOrWhiteSpace($outbase)

if ( ( -not $haveVideoDir -and -not $haveVideoFile ) -or `
     ( $haveVideoDir -and $haveVideoFile ) ) {
    throw "Must specify either -videodir or -videofile"

} elseif ( $haveVideodir ) {
    # Search for all video files in directory
    $videofiles = $( get-childitem  $videodir *$videoext )
    if ( -not $( $videofiles.count -gt 0 ) ) {
        throw "Did not find any video files"
    }
    # Set outbase if necessary
    if ( -not $haveOutbase ) {
        $outbase=$videodir
    }

} elseif ( $haveVideoFile ) {
    # Convert to file object
    $videofileObj = Get-Item $videofile
    # Just set videofiles to input file
    $videofiles=$videofileObj
    # Set outbase if necessary
    if ( -not $haveOutbase) {
        $outbase=$videofileObj.Directory
    }
}

foreach ( $file in  $videofiles ) {
    Write-Host "`n`n$($file.name)`n"
    # Set output directory
    $outdir = join-path $outbase $( $file.basename + "_frames" )

    # Create output directory if it doesn't already exist, pipe to null to
    # suppress printed output
    if ( -not $( test-path $outdir ) ) {
        new-item $outdir -type directory | out-null
    }

    # Run ffmpeg to split frames out
    ffmpeg.exe -y -i $file.fullname -f image2 $( join-path $outdir frame%06d.png )
}

