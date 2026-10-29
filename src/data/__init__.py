from .datasets import BinaryChangeDetectionDataset

# The COG/VRT helpers need the GDAL Python bindings (osgeo), which only the data
# preparation uses; import them on first access so training and inference run without GDAL.
_PROCESSING_UTILS = {"create_clipped_cog_with_bbox", "create_vrt_from_tiles", "create_cog", "read_bbox_from_gpkg"}


def __getattr__(name):
    if name in _PROCESSING_UTILS:
        from .processing import processing_utils

        return getattr(processing_utils, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["BinaryChangeDetectionDataset", "create_clipped_cog_with_bbox", "create_vrt_from_tiles", "create_cog", "read_bbox_from_gpkg"]
