#!/bin/bash
# video.sh - Batch video processing script
# =============================================================================
# Configuration
# =============================================================================

VIDEO_DIR="/mnt/zone/B/Sayak/cage_dataset/Video"
OUTPUT_DIR="/mnt/zone/B/Sayak/processed_videos"

# Temporal smoothing configuration
ENABLE_TEMPORAL_SMOOTHING=true
SMOOTHING_METHOD="gaussian"  # Options: "median", "gaussian", "kalman"
SMOOTHING_WINDOW=9

# =============================================================================
# Functions
# =============================================================================

print_header() {
    echo "=================================================="
    echo "🎬 BATCH VIDEO PROCESSING PIPELINE"
    echo "=================================================="
    echo "📁 Input Directory: $VIDEO_DIR"
    echo "📁 Output Directory: $OUTPUT_DIR"
    echo "🔄 Temporal Smoothing: $([ "$ENABLE_TEMPORAL_SMOOTHING" = true ] && echo "✅ Enabled ($SMOOTHING_METHOD, window=$SMOOTHING_WINDOW)" || echo "❌ Disabled")"
    echo "📺 Terminal Output: ✅ Real-time visible"
    echo "📝 Logging: ❌ Disabled"
    echo "=================================================="
    echo ""
}

validate_environment() {
    echo "🔍 Validating environment..."
    
    # Check if input directory exists
    if [ ! -d "$VIDEO_DIR" ]; then
        echo "❌ Error: Input directory not found: $VIDEO_DIR"
        exit 1
    fi
    
    # Check if frame.py exists
    if [ ! -f "frame.py" ]; then
        echo "❌ Error: frame.py not found in current directory"
        exit 1
    fi
    
    # Check for video files
    VIDEO_COUNT=$(ls -1 "$VIDEO_DIR"/*.mp4 2>/dev/null | wc -l)
    if [ "$VIDEO_COUNT" -eq 0 ]; then
        echo "❌ Error: No MP4 files found in $VIDEO_DIR"
        exit 1
    fi
    
    echo "✅ Environment validation passed"
    echo "📊 Found $VIDEO_COUNT video files to process"
    echo ""
}

process_video() {
    local video_path="$1"
    local video_filename=$(basename "$video_path" .mp4)
    local output_path="$OUTPUT_DIR/processed_$video_filename.mp4"
    
    echo "🎬 Processing: $video_filename"
    echo "📄 Input: $video_path"
    echo "📄 Output: $output_path"
    echo "⏰ Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Build processing command
    local cmd="python frame.py process \"$video_path\" --output \"$output_path\""
    
    # Add temporal smoothing parameters if enabled
    if [ "$ENABLE_TEMPORAL_SMOOTHING" = true ]; then
        cmd="$cmd --temporal-smoothing --smoothing-method \"$SMOOTHING_METHOD\" --smoothing-window $SMOOTHING_WINDOW"
    fi
    
    echo "🚀 Executing: $cmd"
    echo ""
    
    # Execute the command with real-time output
    local start_time=$(date +%s)
    if eval "$cmd"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local duration_min=$((duration / 60))
        local duration_sec=$((duration % 60))
        
        echo ""
        echo "✅ SUCCESS: $video_filename completed in ${duration_min}m ${duration_sec}s"
        
        # Verify output file exists and show size
        if [ -f "$output_path" ]; then
            local file_size=$(stat -f%z "$output_path" 2>/dev/null || stat -c%s "$output_path" 2>/dev/null || echo "0")
            local size_mb=$((file_size / 1024 / 1024))
            echo "📊 Output file size: ${size_mb}MB"
        else
            echo "⚠️ Warning: Output file not found: $output_path"
        fi
        
        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local duration_min=$((duration / 60))
        local duration_sec=$((duration % 60))
        
        echo ""
        echo "❌ FAILED: $video_filename failed after ${duration_min}m ${duration_sec}s"
        return 1
    fi
}

# =============================================================================
# Main Processing
# =============================================================================

main() {
    print_header
    validate_environment
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    echo "📁 Created output directory: $OUTPUT_DIR"
    echo ""
    
    local success_count=0
    local failed_count=0
    local total_start_time=$(date +%s)
    
    echo "🔄 Starting video processing..."
    echo ""
    
    # Process each video file
    for video in "$VIDEO_DIR"/*.mp4; do
        if [ -f "$video" ]; then
            echo "▶️ =========================================="
            
            if process_video "$video"; then
                ((success_count++))
            else
                ((failed_count++))
            fi
            
            echo "=========================================="
            echo ""
        fi
    done
    
    # Calculate total processing time
    local total_end_time=$(date +%s)
    local total_duration=$((total_end_time - total_start_time))
    local total_minutes=$((total_duration / 60))
    local total_seconds=$((total_duration % 60))
    
    # Print final summary
    echo ""
    echo "🎉 =============================================="
    echo "🎉 BATCH PROCESSING COMPLETED!"
    echo "🎉 =============================================="
    echo ""
    echo "📊 Final Summary:"
    echo "   • Total videos processed: $((success_count + failed_count))"
    echo "   • Successfully completed: $success_count"
    echo "   • Failed: $failed_count"
    echo "   • Total processing time: ${total_minutes}m ${total_seconds}s"
    echo "   • Average time per video: $(((total_duration + success_count/2) / success_count))s" 2>/dev/null || echo "   • Average time per video: N/A"
    echo ""
    echo "🔧 Configuration Used:"
    echo "   • Temporal smoothing: $([ "$ENABLE_TEMPORAL_SMOOTHING" = true ] && echo "✅ $SMOOTHING_METHOD (window=$SMOOTHING_WINDOW)" || echo "❌ Disabled")"
    echo "   • Input directory: $VIDEO_DIR"
    echo "   • Output directory: $OUTPUT_DIR"
    echo ""
    
    if [ $success_count -gt 0 ]; then
        echo "📁 Processed videos saved to: $OUTPUT_DIR"
        echo "📋 Output files:"
        ls -la "$OUTPUT_DIR"/*.mp4 2>/dev/null | while read -r line; do
            echo "   $line"
        done
    fi
    
    if [ $failed_count -gt 0 ]; then
        echo ""
        echo "⚠️ WARNING: $failed_count video(s) failed to process"
        echo "   Check the terminal output above for error details"
        exit 1
    else
        echo ""
        echo "🎉 All videos processed successfully!"
    fi
}

# Run main function
main "$@"
