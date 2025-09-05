# Contour Encoder

Um encoder de contornos em Rust que processa imagens PNG e gera representações compactas usando chain codes e simplificação de contornos.

## Funcionalidades

- Detecção de contornos usando algoritmo Suzuki-Abe
- Simplificação de contornos com Ramer-Douglas-Peucker  
- Compressão de chain codes
- Serialização em formato .pot
- Processamento em lote de diretórios

## Pré-requisitos

### Instalar Rust

1. Acesse [https://rustup.rs/](https://rustup.rs/)
2. Execute o comando de instalação:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```
3. Reinicie o terminal ou execute:
   ```bash
   source ~/.cargo/env
   ```

### Instalar FFmpeg (opcional)

Para extrair frames de vídeos:
- **Ubuntu/Debian**: `sudo apt install ffmpeg`
- **macOS**: `brew install ffmpeg`  
- **Windows**: Baixe de [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)

## Compilação

1. Clone ou baixe o código fonte
2. Navegue até o diretório `encoder/`
3. Compile o projeto:
   ```bash
   cargo build --release
   ```

O executável será criado em `target/release/encoder`

## Uso

### Extraindo frames de vídeo

Para extrair frames de um vídeo (ex: Bad Apple):

```bash
# Crie o diretório para os frames
mkdir frames

# Extraia os frames binarizados
ffmpeg -i ./bad-apple.mp4 \
-vf "scale=-1:135,format=gray,geq='if(gt(lum(X,Y),128),255,0)'" frames/frame_%04d.png 
```

### Processando as imagens

Execute o encoder no diretório com as imagens PNG:

```bash
./target/release/encoder frames/
```

Ou se usando `cargo run`:

```bash
cargo run --release -- frames/
```

### Saída

O programa irá:
1. Processar todas as imagens PNG no diretório
2. Redimensionar para 180x135 pixels
3. Detectar e simplificar contornos
4. Gerar um arquivo `.pot` com os dados comprimidos

Exemplo de saída:
```
frames_contours.pot
```

## Configuração

O código possui algumas constantes configuráveis em `src/main.rs`:

- `EPSILON: f64 = 3.0` - Precisão da simplificação Douglas-Peucker
- `target_width: u32 = 180` - Largura alvo das imagens
- `target_height: u32 = 135` - Altura alvo das imagens

## Dependências

O projeto usa as seguintes bibliotecas:
- `image` - Processamento de imagens
- `pot` - Serialização binária
- `serde` - Serialização/deserialização
