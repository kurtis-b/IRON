import csv

from iron.applications.transformer_layer.roofline import annotate_results_csv


def test_annotate_results_csv_writes_peak_and_roofline_fields(tmp_path):
    input_csv = tmp_path / "suite.csv"
    peak_json = tmp_path / "peak.json"
    output_csv = tmp_path / "annotated.csv"
    input_csv.write_text(
        "study_id,backend,execution_mode,pattern_label,seq_len,batch_size,dtype,use_bias,weights_source,source_model_name,source_layer_index,warmup_runs,runs_per_sample,measured_inference_count,timed_total_sec,avg_latency_ms,throughput_flops_per_sec,estimated_flops_per_inference,estimated_bytes_per_inference,operational_intensity_flops_per_byte,backend_peak_ops_per_sec,roofline_bound_ops_per_sec,backend_pct_of_peak,roofline_pct,avg_power_w,max_power_w,energy_j,power_sample_count\n"
        "synthetic_transformer_layer,npu,encoder_pipeline,encoder_pipeline,64,1,bfloat16,False,synthetic,,,1,1,1,1.0,1.0,20.0,10.0,5.0,2.0,,,,,,,,\n",
        encoding="utf-8",
    )
    peak_json.write_text(
        '{"artifact_version":"1","references":[{"backend":"npu","peak_ops_per_sec":40.0,"peak_bytes_per_sec":8.0,"source_note":"measured"}]}',
        encoding="utf-8",
    )

    rows = annotate_results_csv(input_csv, peak_json, output_csv)

    assert len(rows) == 1
    assert float(rows[0]["backend_peak_ops_per_sec"]) == 40.0
    assert float(rows[0]["roofline_bound_ops_per_sec"]) == 16.0
    assert float(rows[0]["backend_pct_of_peak"]) == 0.5
    assert float(rows[0]["roofline_pct"]) == 1.25
    with output_csv.open("r", newline="", encoding="utf-8") as handle:
        written_rows = list(csv.DictReader(handle))
    assert len(written_rows) == 1
    assert float(written_rows[0]["backend_pct_of_peak"]) == 0.5
