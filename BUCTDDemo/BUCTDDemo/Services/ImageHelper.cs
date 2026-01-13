using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace BUCTDDemo.Services;

public static class ImageHelper
{
    /// <summary>
    /// Reads an image stream and returns a 3D byte array [channels, height, width] in RGBA order.
    /// </summary>
    public static async Task<byte[,,]> To3DArrayAsync(Stream stream)
    {
        stream.Position = 0;
        using var image = await Image.LoadAsync<Rgba32>(stream);
        var width = image.Width;
        var height = image.Height;
        const int channels = 4;
        var arr = new byte[channels, height, width];

        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                var px = image[x, y];
                arr[0, y, x] = px.R;
                arr[1, y, x] = px.G;
                arr[2, y, x] = px.B;
                arr[3, y, x] = px.A;
            }
        }

        return arr;
    }
}