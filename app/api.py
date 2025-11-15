import json
import asyncio
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse

from .schemas import InputText
from . import gen_service

router = APIRouter()

@router.post("/predict")
async def generate(input: InputText, request: Request):
    """
    Non-streaming for check endpoint.
    """
    model = request.app.state.model
    tokenizer = request.app.state.tokenizer

    try:
        final_answer = await asyncio.to_thread(
            gen_service.generate_text_non_stream,
            model, 
            tokenizer, 
            input.text
        )
        return {"text": input.text, "generate": final_answer}
    except Exception as e:
        print(f"Error di /predict: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stream-sse")
async def generate_stream_sse(input: InputText, request: Request):
    """
    Streaming Server-Sent Events (SSE) endpoint.
    """
    model = request.app.state.model
    tokenizer = request.app.state.tokenizer

    try:
        stream_generator = gen_service.generate_text_stream(
            model, tokenizer, input.text
        )

        async def sse_generator():
            print("\nOpening the SSE generator wrapper...")
            try:
                async for chunk in stream_generator:
                    if chunk.startswith("[ERROR]"):
                        print(f"Sending SSE error: {chunk}")
                        data = json.dumps({"token": chunk})
                        yield f"data: {data}\n\n"
                        break
                    
                    data = json.dumps({"token": chunk})
                    yield f"data: {data}\n\n"
                
                data_done = json.dumps({"token": "[DONE]"})
                yield f"data: {data_done}\n\n"

            except Exception as e:
                print(f"Error in SSE generator wrapper: {e}")
                data_err = json.dumps({"token": f"[ERROR] {e}"})
                yield f"data: {data_err}\n\n"
            finally:
                print("SSE generator wrapper closed.")
        
        return StreamingResponse(sse_generator(), media_type="text/event-stream")
    
    except Exception as e:
        print(f"Error in endpoint /stream-sse: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to start stream: {e}"
        )