import pandas as pd
from ..analyzers.multi_asset_capm_analyzer import MultiAssetCAPMAnalyzer

class MultiAssetReporter:
    """
    Reporter para generar informes de análisis CAPM multi-activo.
    
    Responsabilidad: Formatear y presentar resultados CAPM para múltiples activos.
    """

    def __init__(self, multi_asset_analyzer: MultiAssetCAPMAnalyzer):
        """
        Args:
            multi_asset_analyzer: Instancia de MultiAssetCAPMAnalyzer
        """
        self.analyzer = multi_asset_analyzer
    
    def generate_summary_report(
        self,
        returns: pd.DataFrame,
        market_returns: pd.Series,
        risk_free_rate: float
    ) -> None:
        """
        Genera reporte resumen de múltiples activos.
        
        Args:
            returns: DataFrame con retornos de múltiples activos
            market_returns: Serie con retornos del mercado
            risk_free_rate: Tasa libre de riesgo anualizada
        """
        analysis = self.analyzer.analyze_multiple(returns, market_returns, risk_free_rate)
        
        if analysis.empty:
            print("⚠️  No hay datos suficientes para el análisis")
            return
        
        # Usar el nivel de significancia del analyzer
        sig_level = self.analyzer.significance_level
        
        print("ANÁLISIS CAPM MULTI-ACTIVO".center(80))

        print(f"Activos analizados: {len(analysis)}")
        
        # Estadísticas generales
        print("ESTADÍSTICAS GENERALES")
        print(f"  Beta promedio:           {analysis['beta'].mean():>8.3f}")
        print(f"  Alpha promedio (anual):  {analysis['alpha_annual'].mean()*100:>8.2f}%")
        print(f"  R² promedio:             {analysis['r_squared'].mean():>8.3f}")
        
        # Alphas significativos
        significant = analysis[analysis['is_significant']]
        print(f"\n[OK] Alphas significativos: {len(significant)} / {len(analysis)}")
        
        # Top performers
        print("TOP 5 PERFORMERS (Alpha más alto)")
        top5 = analysis.nlargest(5, 'alpha_annual')
        for asset, row in top5.iterrows():
            sig = "[OK]" if row['is_significant'] else "    "
            print(f"  {sig} {asset:<10} Alpha: {row['alpha_annual']*100:>7.2f}%  Beta: {row['beta']:>6.3f}")
        
        # Worst performers
        print("BOTTOM 5 PERFORMERS (Alpha más bajo)")
        bottom5 = analysis.nsmallest(5, 'alpha_annual')
        for asset, row in bottom5.iterrows():
            sig = "[OK]" if row['is_significant'] else "    "
            print(f"  {sig} {asset:<10} Alpha: {row['alpha_annual']*100:>7.2f}%  Beta: {row['beta']:>6.3f}")
    
        print(f"Nota: [OK] indica alpha estadísticamente significativo (p < {sig_level})")