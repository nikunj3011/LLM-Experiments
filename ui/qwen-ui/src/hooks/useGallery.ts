import { useCallback, useEffect, useState } from "react";
import { api } from "../api";
import type { GalleryAsset } from "../types";

export function useGallery(kind: "image" | "video") {
  const [assets, setAssets] = useState<GalleryAsset[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  const refresh = useCallback(async () => {
    setIsLoading(true);
    try {
      const gallery = await api.gallery();
      setAssets(kind === "image" ? gallery.images ?? [] : gallery.videos ?? []);
    } finally {
      setIsLoading(false);
    }
  }, [kind]);

  useEffect(() => { void refresh(); }, [refresh]);
  return { assets, isLoading, refresh };
}
