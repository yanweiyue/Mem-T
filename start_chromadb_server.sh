HOST=${CHROMA_HOST:-"localhost"}
PORT=${CHROMA_PORT:-8070}
DATA_PATH=${CHROMA_PATH:-"./database/locomo"}

echo "================================"
echo "Starting ChromaDB HTTP server"
echo "================================"
echo "Host: $HOST"
echo "Port: $PORT"
echo "Data Path: $DATA_PATH"
echo "================================"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""


chroma run --host $HOST --port $PORT --path $DATA_PATH




