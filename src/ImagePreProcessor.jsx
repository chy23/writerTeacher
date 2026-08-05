import React, { useState, useRef, useEffect } from 'react';
import { Settings2, RotateCw, Crop, Check, X, Image as ImageIcon } from 'lucide-react';

export default function ImagePreProcessor({ file, onComplete, onCancel }) {
  const canvasRef = useRef(null);
  const [image, setImage] = useState(null);
  
  // Settings state
  const [brightness, setBrightness] = useState(100);
  const [contrast, setContrast] = useState(100);
  const [rotation, setRotation] = useState(0);
  
  // Crop state (percentages 0-100)
  const [crop, setCrop] = useState({ top: 0, bottom: 0, left: 0, right: 0 });

  useEffect(() => {
    if (!file) return;
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      setImage(img);
      URL.revokeObjectURL(url);
    };
    img.src = url;
  }, [file]);

  useEffect(() => {
    if (!image || !canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // Calculate final dimensions after crop
    const cw = image.width * (1 - (crop.left + crop.right) / 100);
    const ch = image.height * (1 - (crop.top + crop.bottom) / 100);
    
    // Set canvas size based on rotation (swap width/height if 90 or 270)
    if (rotation === 90 || rotation === 270) {
      canvas.width = ch;
      canvas.height = cw;
    } else {
      canvas.width = cw;
      canvas.height = ch;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Apply filters
    ctx.filter = `brightness(${brightness}%) contrast(${contrast}%)`;
    
    // Handle rotation and translate context to center
    ctx.translate(canvas.width / 2, canvas.height / 2);
    ctx.rotate((rotation * Math.PI) / 180);
    
    // Source crop coords
    const sx = image.width * (crop.left / 100);
    const sy = image.height * (crop.top / 100);
    const sw = cw;
    const sh = ch;
    
    // Draw image centered in the rotated context
    ctx.drawImage(
      image,
      sx, sy, sw, sh,
      -cw / 2, -ch / 2, cw, ch
    );
    
  }, [image, brightness, contrast, rotation, crop]);

  const handleApply = () => {
    if (!canvasRef.current) return;
    canvasRef.current.toBlob(blob => {
      const processedFile = new File([blob], file.name, { type: 'image/jpeg' });
      onComplete(processedFile);
    }, 'image/jpeg', 0.85);
  };

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
      <div className="bg-white rounded-3xl w-full max-w-5xl flex flex-col md:flex-row overflow-hidden shadow-2xl animate-in zoom-in-95 duration-300">
        
        {/* Preview Area */}
        <div className="flex-1 bg-slate-900 flex items-center justify-center p-4 min-h-[400px] relative overflow-hidden">
          <canvas 
            ref={canvasRef} 
            className="max-w-full max-h-[70vh] object-contain rounded shadow-lg border border-white/10"
          />
        </div>

        {/* Controls Area */}
        <div className="w-full md:w-80 bg-white p-6 flex flex-col max-h-[80vh] overflow-y-auto">
          <h2 className="text-xl font-black text-slate-800 mb-6 flex items-center gap-2">
            <Settings2 className="text-indigo-600" />
            圖片預處理
          </h2>

          <div className="space-y-6 flex-1">
            {/* Rotate */}
            <div>
              <label className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-3 block">旋轉照片</label>
              <button 
                onClick={() => setRotation(r => (r + 90) % 360)}
                className="w-full py-2.5 bg-indigo-50 text-indigo-700 rounded-xl font-bold hover:bg-indigo-100 flex items-center justify-center gap-2 transition-colors"
              >
                <RotateCw size={18} /> 向右旋轉 90°
              </button>
            </div>

            <hr className="border-slate-100" />

            {/* Adjustments */}
            <div>
              <label className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-3 flex items-center gap-1"><ImageIcon size={14} /> 對比與亮度</label>
              
              <div className="mb-4">
                <div className="flex justify-between text-xs mb-1 font-bold text-slate-600">
                  <span>亮度 (Brightness)</span>
                  <span>{brightness}%</span>
                </div>
                <input type="range" min="50" max="200" value={brightness} onChange={e => setBrightness(Number(e.target.value))} className="w-full accent-indigo-600" />
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1 font-bold text-slate-600">
                  <span>對比 (Contrast)</span>
                  <span>{contrast}%</span>
                </div>
                <input type="range" min="50" max="200" value={contrast} onChange={e => setContrast(Number(e.target.value))} className="w-full accent-indigo-600" />
              </div>
            </div>

            <hr className="border-slate-100" />

            {/* Crop */}
            <div>
              <label className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-3 flex items-center gap-1"><Crop size={14} /> 邊緣裁切 (%)</label>
              
              <div className="space-y-3">
                {Object.keys(crop).map(dir => (
                  <div key={dir} className="flex items-center gap-2">
                    <span className="text-xs font-bold text-slate-600 w-10 text-right capitalize">{
                      dir === 'top' ? '上' : dir === 'bottom' ? '下' : dir === 'left' ? '左' : '右'
                    }</span>
                    <input 
                      type="range" min="0" max="40" value={crop[dir]} 
                      onChange={e => setCrop({...crop, [dir]: Number(e.target.value)})} 
                      className="flex-1 accent-rose-500" 
                    />
                    <span className="text-xs font-bold text-slate-400 w-8">{crop[dir]}%</span>
                  </div>
                ))}
              </div>
              <p className="text-[10px] text-slate-400 mt-2 leading-tight">如果邊緣有拍到桌面或其他考卷，請使用拉桿將其裁切，以提升 AI 辨識準確率。</p>
            </div>
          </div>

          <div className="mt-8 pt-4 border-t border-slate-100 flex gap-3">
            <button 
              onClick={onCancel}
              className="flex-1 py-3 bg-slate-100 text-slate-600 rounded-xl font-bold hover:bg-slate-200 transition-colors flex items-center justify-center gap-2"
            >
              <X size={18} /> 取消
            </button>
            <button 
              onClick={handleApply}
              className="flex-1 py-3 bg-indigo-600 text-white rounded-xl font-bold hover:bg-indigo-700 transition-colors flex items-center justify-center gap-2 shadow-lg shadow-indigo-200"
            >
              <Check size={18} /> 確認套用
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
