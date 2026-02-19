# Avatar Reference Images

Each avatar needs a directory under `avatars/` containing a reference portrait image.

## Directory Structure

```
avatars/
├── default/
│   └── reference.png    # 512x512 front-facing portrait
├── salem/
│   └── reference.png
└── fatima/
    └── reference.png
```

## Image Requirements

| Property | Requirement |
|----------|-------------|
| Format | PNG, JPG, JPEG, or WebP |
| Resolution | 512x512 pixels (recommended) |
| Content | Front-facing portrait, neutral expression |
| Background | Solid or simple background preferred |
| Lighting | Even, no harsh shadows |

## File Naming

The resolver looks for files in this priority order:

1. `reference.png` / `reference.jpg` / `reference.jpeg` / `reference.webp`
2. `avatar.png` / `avatar.jpg` / `avatar.jpeg` / `avatar.webp`
3. `<avatar_id>.png` / `<avatar_id>.jpg` / etc.
4. First image file found in the directory

## Usage

Specify the avatar by its directory name in API calls:

```json
{
  "text": "Hello",
  "avatar_id": "default"
}
```

## Default Avatar

The included `default/reference.png` is a placeholder silhouette. Replace it with a real portrait photo for production use with EchoMimicV2 video generation.
